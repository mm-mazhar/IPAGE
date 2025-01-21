from sqlalchemy.orm import class_mapper


def object_to_dict(obj):
    # Extract columns from the SQLAlchemy model
    columns = [c.key for c in class_mapper(obj.__class__).columns]
    return {column: getattr(obj, column) for column in columns}
