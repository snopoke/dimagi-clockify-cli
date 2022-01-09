from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

import httpx
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from dimagi_clockify_cli.config import Config
from dimagi_clockify_cli.db import Project, Tag, Task, User, Workspace


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
) -> Project:
    stmt = (
        select(Project)
        .options(selectinload(Project.workspace))
        .where(Project.name == project_name)
    )
    result = await session.execute(stmt)
    project = result.scalars().one_or_none()
    if project is None:
        project = await fetch_project(session, client, workspace, project_name)
    return project


async def fetch_project(
        session,
        client: Client,
        workspace: Workspace,
        project_name: str,
) -> Project:
    endpoint = f'/workspaces/{workspace.id}/projects'
    params = {'name': project_name}
    response = await client.request('GET', endpoint, params=params)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f'Project "{project_name}" not found')
    if len(data) > 1:
        raise ValueError(f'Multiple projects "{project_name}" found')
    project = Project(
        id=data[0]['id'],
        # Use `project_name` instead of `data[0]['name']`, because if
        # they are not exactly the same, we will keep fetching.
        name=project_name,
        workspace=workspace,
    )
    session.add(project)
    await session.commit()
    return project


async def get_tags(
        session,
        client: Client,
        workspace: Workspace,
        tag_names: List[str]) -> List[Tag]:
    tags = []
    for tag_name in tag_names:
        stmt = (
            select(Tag)
            .options(selectinload(Tag.workspace))
            .where(Tag.name == tag_name)
        )
        result = await session.execute(stmt)
        tag = result.scalars().one_or_none()
        if tag is None:
            tag = await fetch_tag(session, client, workspace, tag_name)
        tags.append(tag)
    return tags


async def fetch_tag(
        session,
        client: Client,
        workspace: Workspace,
        tag_name: str,
) -> Tag:
    endpoint = f'/workspaces/{workspace.id}/tags'
    params = {
        'name': tag_name,
        'archived': False,
    }
    response = await client.request('GET', endpoint, params=params)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f'Tag "{tag_name}" not found')
    if len(data) > 1:
        raise ValueError(f'Multiple tags "{tag_name}" found')
    tag = Tag(
        id=data[0]['id'],
        name=tag_name,
        workspace=workspace,
    )
    session.add(tag)
    await session.commit()
    return tag


async def get_task(
        session,
        client: Client,
        project: Project,
        task_name: str,
) -> Task:
    stmt = (
        select(Task)
        .options(selectinload(Task.project).selectinload(Project.workspace))
        .where(Task.project == project)
        .where(Task.name == task_name)
    )
    result = await session.execute(stmt)
    task = result.scalars().one_or_none()
    if task is None:
        task = await fetch_task(session, client, project, task_name)
    return task


async def fetch_task(
        session,
        client: Client,
        project: Project,
        task_name: str,
) -> Task:
    endpoint = (
        f'/workspaces/{project.workspace.id}'
        f'/projects/{project.id}/tasks'
    )
    params = {'name': task_name}
    response = await client.request('GET', endpoint, params=params)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f'Task "{task_name}" not found')
    if len(data) > 1:
        raise ValueError(f'Multiple tasks "{task_name}" found')
    task = Task(
        id=data[0]['id'],
        name=task_name,
        project=project,
    )
    session.add(task)
    await session.commit()
    return task


async def stop_timer(
        client: Client,
        workspace: Workspace,
        user: User,
        since_dt: Optional[datetime] = None,
):
    if since_dt is None:
        since_dt = datetime.utcnow()
    endpoint = f'/workspaces/{workspace.id}/user/{user.id}/time-entries'
    # End a second earlier to prevent overlapping time entries
    body = {'end': zulu(since_dt - timedelta(seconds=1))}
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
):
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
    response = await client.request('POST', endpoint, json=body)
    response.raise_for_status()


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


def zulu(utc_dt: datetime) -> str:
    return utc_dt.isoformat(sep='T', timespec='seconds') + 'Z'


def is_billable(tags: List[Tag]) -> bool:
    return not any(t.name == 'Overhead' for t in tags)
