import asyncio
import csv
import sys
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, date, timedelta
from operator import itemgetter
from typing import Optional

import typer
import yaml
from pydantic.error_wrappers import ValidationError

from dimagi_clockify_cli.bootstrap import bootstrap_db, check_config
from dimagi_clockify_cli.config import get_config, Config, Bucket
from dimagi_clockify_cli.db import get_session, init_db
from dimagi_clockify_cli.services import (
    add_time_entry,
    get_client,
    get_project,
    get_tags,
    get_task,
    get_user,
    get_workspace,
    stop_timer,
    get_buckets,
)
from dimagi_clockify_cli.utils import parse_date, parse_hours, get_input

app = typer.Typer()


@app.command()
def dcl(
        ctx: typer.Context,
        bucket: str,
        since: Optional[str] = None,
):
    config = ctx.obj['config']
    if bucket == 'stop':
        asyncio.run(stop(config))
    elif bucket == 'list':
        bucket_list = sorted(config.buckets.keys())
        typer.echo('\n'.join(bucket_list))
    elif bucket == 'bootstrap_db':
        asyncio.run(bootstrap(config))
    elif bucket == 'check_config':
        asyncio.run(check_config_(config))
    elif bucket == 'list_entries':
        asyncio.run(list_entries_(config))
    else:
        if bucket not in config.buckets:
            typer.echo(f'Unknown bucket "{bucket}"')
            raise typer.Exit()
        bucket_obj = config.buckets[bucket]
        asyncio.run(work_on(config, bucket_obj, since))


@app.callback()
def init(ctx: typer.Context):
    logging.basicConfig(level=logging.INFO)
    try:
        config = get_config()
    except ValidationError as err:
        typer.echo(f'Error loading config: {err}')
        raise typer.Exit()
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@app.command("from-csv")
def dcl_csv(ctx: typer.Context, path: str, bulk: bool = False):
    """
    Create time report entries from a CSV file.

    CSV format:

        bucket,2022-05-01,2022-05-02\n
        email,1,0.5\n
        code_review,0.5,\n

    Rows with empty date or bucket are ignored.
    """
    from_csv(ctx.obj['config'], path, bulk)


async def bootstrap(config: Config):
    await init_db()
    async with get_session() as session, \
            get_client(config) as client:
        await bootstrap_db(session, client)


async def list_entries_(config: Config):
    async with get_session() as session, \
            get_client(config) as client:
        user = await get_user(session, client)
        entries = await get_buckets(session, client, user)
        yaml.dump([e.dict() for e in entries], sys.stdout)


async def check_config_(config: Config):
    async with get_session() as session, \
            get_client(config) as client:
        await check_config(session, client, config)


async def work_on(
        config: Config,
        bucket: Bucket,
        since: Optional[str] = None,
):
    if since is None:
        since_dt = datetime.now()
    else:
        since_t = time.fromisoformat(since)
        since_dt = datetime.combine(date.today(), since_t)
    await create_time_entry(config, bucket, since_dt)


async def create_time_entry(
        config,
        bucket,
        since_dt: Optional[datetime] = None,
        until_dt: Optional[datetime] = None):
    await init_db()
    async with get_session() as session, \
            get_client(config) as client:
        workspace = await get_workspace(session, client)
        project = await get_project(session, client, workspace, bucket.project)
        tags = await get_tags(session, client, workspace, bucket.tags)
        user = await get_user(session, client)
        task = await get_task(session, client, project, bucket.task)
        await stop_timer(client, workspace, user, since_dt)
        await add_time_entry(
            client,
            bucket.description,
            workspace,
            project,
            task,
            tags,
            since_dt,
            until_dt
        )


async def stop(config: Config):
    async with get_session() as session, \
            get_client(config) as client:
        workspace = await get_workspace(session, client)
        user = await get_user(session, client)
        await stop_timer(client, workspace, user)


def from_csv(config, path, bulk):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        records_by_date = get_records(config, list(reader))

    all_records = []
    for date_, records in sorted(records_by_date.items(), key=itemgetter(0)):
        total_hours = sum(record.hours for record in records)
        typer.echo(f"{date_} - {total_hours} total hours")
        for record in records:
            typer.echo(
                f"  Creating Entry: '{record.bucket.description}' - "
                f"{record.hours} hour(s) from {record.start.time()} to {record.end.time()}"
            )

        if bulk:
            all_records.extend(records)
        else:
            create_entries_for_records(config, records)

    if all_records:
        create_entries_for_records(config, all_records)


def create_entries_for_records(config, records):
    input_ = get_input("Create entries [y\\n\\q]: ")
    if input_ == 'n':
        typer.echo("  skip")
    if input_ == 'q':
        raise typer.Exit()
    if input_ != 'y':
        typer.echo('Unknown input')
        raise typer.Exit()

    async def create():
        for record in records:
            await create_time_entry(config, record.bucket, record.start, record.end)

    asyncio.run(create())


@dataclass
class Record:
    bucket: Bucket
    date: date
    hours: float
    start: datetime
    end: datetime


def get_records(config, csv_rows):
    time_by_date = {}

    def get_start(date_):
        default = datetime(
            year=date_.year, month=date_.month, day=date_.day,
            hour=9
        )
        return time_by_date.setdefault(date, default)

    records_by_date = defaultdict(list)
    for row in csv_rows:
        try:
            bucket_name = row.pop('bucket')
            if not bucket_name:
                continue
            if bucket_name not in config.buckets:
                typer.echo(f'Unknown bucket "{bucket_name}"', err=True)
                raise typer.Exit()
            bucket = config.buckets[bucket_name]
            for date_str, hours_str in row.items():
                if not hours_str:
                    continue
                date = parse_date(date_str)
                hours = parse_hours(hours_str)
                start_time = get_start(date)
                end_time = start_time + timedelta(hours=hours)
                records_by_date[date].append(
                    Record(bucket, date, hours, start_time, end_time)
                )
                time_by_date[date] = end_time
        except Exception as e:
            typer.echo(str(e), err=True)
            raise typer.Exit()

    return records_by_date


if __name__ == "__main__":
    app()
