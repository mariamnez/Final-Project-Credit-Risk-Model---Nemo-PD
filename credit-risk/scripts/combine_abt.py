import duckdb

con = duckdb.connect()

# combine all yearly ABTs into one file
con.execute("""
COPY (SELECT * FROM 'data/processed/abt_*.parquet')
TO 'data/processed/abt.parquet' (FORMAT PARQUET);
""")
print("Combined -> data/processed/abt.parquet")

# quick checks
print(con.execute("SELECT COUNT(*) AS rows FROM 'data/processed/abt.parquet'").df())
print(con.execute("""
SELECT vintage_q, COUNT(*) AS n,
       AVG(CASE WHEN default_within_24m THEN 1 ELSE 0 END)::DOUBLE AS bad_rate
FROM 'data/processed/abt.parquet'
GROUP BY 1 ORDER BY 1
""").df().head(12))
