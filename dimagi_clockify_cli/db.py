import os
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Field, Relationship, SQLModel

from dimagi_clockify_cli.config import get_config_dir


class Workspace(SQLModel, table=True):
    id: str = Field(primary_key=True)


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    workspace_id: str = Field(foreign_key='workspace.id')
    workspace: Workspace = Relationship()


class Project(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    workspace_id: str = Field(foreign_key='workspace.id')
    workspace: Workspace = Relationship()


class Task(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    project_id: str = Field(foreign_key='project.id')
    project: Project = Relationship()


class Tag(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    workspace_id: str = Field(foreign_key='workspace.id')
    workspace: Workspace = Relationship()


def get_engine():
    config_dir = get_config_dir()
    filename = os.path.join(config_dir, 'cache.db')
    return create_async_engine(f'sqlite+aiosqlite:///{filename}')


@asynccontextmanager
async def get_session():
    engine = get_engine()
    session_class = sessionmaker(
        engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    async with session_class() as session:
        yield session


async def init_db():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
