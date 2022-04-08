import asyncio
from datetime import datetime, time, date
from typing import Optional

import typer
from pydantic.error_wrappers import ValidationError

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
)

app = typer.Typer()


@app.command()
def dcl(bucket: str, since: Optional[str] = None):
    try:
        config = get_config()
    except ValidationError as err:
        typer.echo(f'Error loading config: {err}')
        raise typer.Exit()
    if bucket == 'stop':
        asyncio.run(stop(config))
    elif bucket == 'list':
        bucket_list = sorted(config.buckets.keys())
        typer.echo('\n'.join(bucket_list))
    else:
        if bucket not in config.buckets:
            typer.echo(f'Unknown bucket "{bucket}"')
            raise typer.Exit()
        bucket_obj = config.buckets[bucket]
        asyncio.run(work_on(config, bucket_obj, since))


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
        )


async def stop(config: Config):
    async with get_session() as session, \
            get_client(config) as client:
        workspace = await get_workspace(session, client)
        user = await get_user(session, client)
        await stop_timer(client, workspace, user)
