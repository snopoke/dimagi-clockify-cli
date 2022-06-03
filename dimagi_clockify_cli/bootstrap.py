from dimagi_clockify_cli.config import Config
from dimagi_clockify_cli.services import fetch_project, fetch_tag, fetch_task, Client, get_user, \
    get_project, get_task, get_tags


async def bootstrap_db(session, client: Client):
    user = await get_user(session, client)
    workspace = user.workspace
    projects = await fetch_project(session, client, workspace)
    await fetch_tag(session, client, workspace)
    for project in projects:
        await fetch_task(session, client, project)


async def check_config(session, client: Client, config: Config):
    user = await get_user(session, client)
    for name, bucket in config.buckets.items():
        print(f"Checking '{name}'", end='')
        errs = await _check_bucket(bucket, client, session, user)
        if not errs:
            print(": OK")
        else:
            print()
            for err in errs:
                print(f"\t{err}")


async def _check_bucket(bucket, client, session, user):
    project = await get_project(session, client, user.workspace, bucket.project, False)
    errs = []
    if not project:
        errs.append(f"Project '{bucket.project}' not found")
        return errs

    task = await get_task(session, client, project, bucket.task, False)
    if not task:
        errs.append(f"Task '{bucket.task}' not found")

    tags = await get_tags(session, client, user.workspace, bucket.tags, False)
    missing = set(bucket.tags) - {t.name for t in tags}
    if missing:
        errs.append(f"Missing tags '{missing}'")

    return errs
