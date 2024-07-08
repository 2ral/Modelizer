from msticpy.data.sql_to_kql import sql_to_kql


def sql2kql(query: str) -> str:
    kql = sql_to_kql(query)
    kql.replace("  not", " not")
    return kql
