import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import List, Optional

import httpx
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from dimagi_clockify_cli.config import Config, Bucket
from dimagi_clockify_cli.db import Project, Tag, Task, User, Workspace

logger = logging.getLogger(__name__)


class Client:
    """
    A thin wrapper around httpx.AsyncClient.
    """

    def __init__(
            self,
            httpx_client: httpx.AsyncClient,
            config: Config,
    ):
        self.httpx_client = httpx_client
        self.config = config

    async def request(
            self,
            method: str,
            endpoint: str,
            **kwargs
    ) -> httpx.Response:
        url = slash_join(self.config.base_url, endpoint)
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Api-Key': self.config.api_key,
        }
        headers.update(kwargs.pop('headers', {}))
        response = await self.httpx_client.request(
            method,
            url,
            headers=headers,
            **kwargs,
        )
        return response


@asynccontextmanager
async def get_client(config):
    async with httpx.AsyncClient() as httpx_client:
        client = Client(httpx_client, config)
        yield client


async def get_workspace(
        session,
        client: Client,
) -> Workspace:
    stmt = select(Workspace)
    result = await session.execute(stmt)
    workspace = result.scalars().one_or_none()
    if workspace is None:
        user = await fetch_user(session, client)
        workspace = user.workspace
    return workspace


async def get_user(
        session,
        client,
) -> User:
    stmt = select(User).options(selectinload(User.workspace))
    result = await session.execute(stmt)
    user = result.scalars().one_or_none()
    if user is None:
        user = await fetch_user(session, client)
    return user


async def fetch_user(
        session,
        client: Client,
) -> User:
    logger.info("Fetching user")
    response = await client.request('GET', '/user')
    response.raise_for_status()
    data = response.json()
    workspace = Workspace(id=data['defaultWorkspace'])
    user = User(id=data['id'], workspace=workspace)
    session.add(user)
    await session.commit()
    return user


async def get_project(
        session,
        client: Client,
        workspace: Workspace,
        project_name: str,
        fetch: bool = True
) -> Project:
    stmt = (
        select(Project)
        .options(selectinload(Project.workspace))
        .where(Project.name == project_name)
    )
    result = await session.execute(stmt)
    project = result.scalars().one_or_none()
    if project is None and fetch:
        with await fetch_project(session, client, workspace, project_name) as projects:
            project = projects[0]
    return project


async def fetch_project(
        session,
        client: Client,
        workspace: Workspace,
        project_name: Optional[str] = None,
) -> List[Project]:
    logger.info("Fetching projects")
    endpoint = f'/workspaces/{workspace.id}/projects'
    params = {'name': project_name} if project_name else {}
    response = await client.request('GET', endpoint, params=params)
    response.raise_for_status()
    data = response.json()
    if project_name:
        if not data:
            raise ValueError(f'Project "{project_name}" not found')
        if len(data) > 1:
            raise ValueError(f'Multiple projects "{project_name}" found')
    result = await session.execute(select(Project))
    project_ids = {p.id for p in result.scalars().all()}
    projects = []
    for project_data in data:
        project = Project(
            id=project_data['id'],
            # Use `project_name` instead of `data[0]['name']`, because if
            # they are not exactly the same, we will keep fetching.
            name=project_name if project_name else project_data['name'],
            workspace=workspace,
        )
        projects.append(project)
        if project.id not in project_ids:
            logger.info("Saving project %s", project.name)
            session.add(project)

    await session.commit()
    return projects


async def get_tags(
        session,
        client: Client,
        workspace: Workspace,
        tag_names: List[str],
        fetch: bool = True) -> List[Tag]:
    tags = []
    for tag_name in tag_names:
        stmt = (
            select(Tag)
            .options(selectinload(Tag.workspace))
            .where(Tag.name == tag_name)
        )
        result = await session.execute(stmt)
        tag = result.scalars().one_or_none()
        if tag is None and fetch:
            with await fetch_tag(session, client, workspace, tag_name) as tags:
                tag = tags[0]
        if tag:
            tags.append(tag)
    return tags


async def fetch_tag(
        session,
        client: Client,
        workspace: Workspace,
        tag_name: Optional[str] = None,
) -> List[Tag]:
    logger.info("Fetching tags")
    endpoint = f'/workspaces/{workspace.id}/tags'
    params = {
        'archived': False,
    }
    if tag_name:
        params['name'] = tag_name
    response = await client.request('GET', endpoint, params=params)
    response.raise_for_status()
    data = response.json()
    if tag_name:
        if not data:
            raise ValueError(f'Tag "{tag_name}" not found')
        if len(data) > 1:
            raise ValueError(f'Multiple tags "{tag_name}" found')

    stmt = select(Tag).options(selectinload(Tag.workspace))
    result = await session.execute(stmt)
    tag_ids = {tag.id for tag in result.scalars().all()}
    tags = []
    for tag_data in data:
        tag = Tag(
            id=tag_data['id'],
            name=tag_name if tag_name else tag_data['name'],
            workspace=workspace,
        )
        if tag.id not in tag_ids:
            logger.info("Saving tag %s", tag.name)
            session.add(tag)
        tags.append(tag)
    await session.commit()
    return tags


async def fetch_tag_by_id(
        session,
        client: Client,
        workspace: Workspace,
        tag_id: str
) -> Tag:
    logger.info("Fetching tag %s", tag_id)
    endpoint = f'/workspaces/{workspace.id}/tags/{tag_id}'
    response = await client.request('GET', endpoint)
    response.raise_for_status()
    data = response.json()
    print(data)
    if not data:
        raise ValueError(f'Tag "{tag_id}" not found')

    tag = Tag(
        id=data['id'],
        name=data['name'],
        workspace=workspace,
    )
    logger.info("Saving tag %s", tag.name)
    session.add(tag)
    await session.commit()
    return tag


async def get_task(session, client: Client, project: Project, task_name: str,
                   fetch: bool = True) -> Task:
    stmt = (
        select(Task)
        .options(selectinload(Task.project).selectinload(Project.workspace))
        .where(Task.project == project)
        .where(Task.name == task_name)
    )
    result = await session.execute(stmt)
    task = result.scalars().one_or_none()
    if task is None and fetch:
        with await fetch_task(session, client, project, task_name) as tasks:
            task = tasks[0]
    return task


async def fetch_task(
        session,
        client: Client,
        project: Project,
        task_name: Optional[str] = None,
) -> List[Task]:
    logger.info("Fetching tasks")
    endpoint = (
        f'/workspaces/{project.workspace.id}'
        f'/projects/{project.id}/tasks'
    )
    params = {}
    if task_name:
        params = {'name': task_name}
    response = await client.request('GET', endpoint, params=params)
    response.raise_for_status()
    data = response.json()
    if task_name:
        if not data:
            raise ValueError(f'Task "{task_name}" not found')
        if len(data) > 1:
            raise ValueError(f'Multiple tasks "{task_name}" found')

    result = await session.execute(select(Task).where(Task.project == project))
    task_ids = {t.id for t in result.scalars().all()}
    tasks = []
    for task_data in data:
        task = Task(
            id=task_data['id'],
            name=task_name if task_name else task_data['name'],
            project_id=project.id,
        )
        if task.id not in task_ids:
            logger.info("Saving task %s", task.name)
            session.add(task)
    await session.commit()
    return tasks


async def get_buckets(session, client, user):
    workspace = user.workspace
    endpoint = f'/workspaces/{user.workspace.id}/user/{user.id}/time-entries'
    response = await client.request('GET', endpoint, params={'page-size': 100})
    response.raise_for_status()
    buckets = set()
    for entry in response.json():
        description = entry['description']
        project_id = entry['projectId']
        task_id = entry['taskId']
        project = await get_object(session, Project, project_id)
        task = await get_object(session, Task, task_id)
        tags = [
            await get_object(session, Tag, tag_id, partial(
                fetch_tag_by_id, session, client, workspace, tag_id
            ))
            for tag_id in entry['tagIds']
        ]
        tag_names = [t.name for t in tags if t]
        buckets.add(Bucket(
            description=description,
            project=project.name,
            task=task.name,
            tags=tag_names))
    return list(buckets)


async def get_object(session, clazz, id: str, fetcher: Optional = None):
    result = await session.execute(select(clazz).where(clazz.id == id))
    obj = result.scalars().one_or_none()
    if not obj and fetcher:
        obj = await fetcher()
    if not obj:
        logger.warning("Unable to fetch %s with id %s", clazz, id)
    return obj


async def stop_timer(
        client: Client,
        workspace: Workspace,
        user: User,
        since_dt: Optional[datetime] = None,
):
    if since_dt is None:
        since_dt = datetime.now()
    endpoint = f'/workspaces/{workspace.id}/user/{user.id}/time-entries'
    # End a minute earlier to prevent overlapping time entries
    body = {'end': zulu(since_dt - timedelta(minutes=1))}
    # Returns a 404 if timer is not running
    await client.request('PATCH', endpoint, json=body)


async def add_time_entry(
        client: Client,
        description: str,
        workspace: Workspace,
        project: Project,
        task: Task,
        tags: List[Tag],
        since_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
) -> str:
    if since_dt is None:
        since_dt = datetime.utcnow()
    endpoint = f'/workspaces/{workspace.id}/time-entries'
    body = {
        'start': zulu(since_dt),
        'billable': is_billable(tags),
        'description': description,
        'projectId': project.id,
        'taskId': task.id,
        'tagIds': [t.id for t in tags]
    }
    if end_dt:
        body['end'] = zulu(end_dt)
    response = await client.request('POST', endpoint, json=body)
    response.raise_for_status()
    return response.json()['id']


def slash_join(*strings) -> str:
    """
    Joins strings with a single ``/``.

    >>> slash_join('http://example.com', 'foo')
    'http://example.com/foo'
    >>> slash_join('http://example.com/', '/foo/')
    'http://example.com/foo/'

    """
    if len(strings) == 0:
        return ''
    if len(strings) == 1:
        return strings[0]
    left = [strings[0].rstrip('/')]
    right = [strings[-1].lstrip('/')]
    middle = [s.strip('/') for s in strings[1:-1]]
    return '/'.join(left + middle + right)


def zulu(local_dt: datetime) -> str:
    """
    Returns a UTC datetime as ISO-formatted in Zulu time.

    >>> dt = datetime.utcfromtimestamp(1640995200)
    >>> zulu(dt)
    '2022-01-01T00:00:00Z'

    """
    utc_dt = local_dt.astimezone(timezone.utc)
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def is_billable(tags: List[Tag]) -> bool:
    """
    Returns ``True`` if any tag in ``tags`` is "Overhead"

    >>> overhead = Tag(name='Overhead')
    >>> not_overhead = Tag(name='Foo:Bar')
    >>> is_billable([not_overhead])
    True
    >>> is_billable([not_overhead, overhead])
    False

    """
    return not any(t.name == 'Overhead' for t in tags)
