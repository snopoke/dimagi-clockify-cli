import asyncio
import sys
from datetime import date

import typer


def parse_date(date_str: str):
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise Exception(f"Unable to parse date: {date_str}")


def parse_hours(hours: str):
    try:
        return float(hours)
    except (TypeError, ValueError):
        raise Exception(f"Unable to parse hours: {hours}")


def get_input(msg: str):
    return input(msg)
