"""Fixture ORM model (db_schema fact parser target)."""


class Pet:
    __tablename__ = "pets"

    id = None
    name = None
    species = None
