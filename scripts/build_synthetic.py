"""Build data/synthetic_questions.json.

15 hand-written questions covering SQL patterns the dev set doesn't exercise.
Gold SQL is run against Chinook here, and the captured rows become
`expected_result` — so the file is always consistent with the live database.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils import load_db, query_db

# (qid, tier, pattern, question, gold_sql)
QUESTIONS: list[tuple[str, int, str, str, str]] = [
    # 1. Subquery in WHERE: comparison to a scalar aggregate.
    ("s_001", 1, "subquery_where",
     "How many tracks are longer than the average track length (in milliseconds)?",
     "SELECT COUNT(*) AS LongerThanAvg FROM Track "
     "WHERE Milliseconds > (SELECT AVG(Milliseconds) FROM Track)"),

    # 2. CASE in SELECT, with GROUP BY on the derived bucket.
    ("s_002", 2, "case_select",
     "Bucket tracks by length: short (< 3 min), medium (3–5 min), long (> 5 min). "
     "How many tracks are in each bucket?",
     "SELECT CASE "
     "  WHEN Milliseconds < 180000 THEN 'short' "
     "  WHEN Milliseconds <= 300000 THEN 'medium' "
     "  ELSE 'long' "
     "END AS LengthBucket, COUNT(*) AS TrackCount "
     "FROM Track GROUP BY LengthBucket ORDER BY TrackCount DESC"),

    # 3. IS NULL on a self-referential FK.
    ("s_003", 1, "null_filter",
     "Which employees do not report to a manager? Show their full name and title.",
     "SELECT FirstName || ' ' || LastName AS EmployeeName, Title "
     "FROM Employee WHERE ReportsTo IS NULL"),

    # 4. Date range filter on an ISO datetime column.
    ("s_004", 2, "date_range",
     "How many invoices were issued in the first quarter of 2022 (January through March)?",
     "SELECT COUNT(*) AS Q1_2022_Invoices FROM Invoice "
     "WHERE InvoiceDate >= '2022-01-01' AND InvoiceDate < '2022-04-01'"),

    # 5. Self-join on Employee for manager hierarchy.
    ("s_005", 2, "self_join",
     "List each employee with their manager's name. Include employees who have "
     "no manager (show the manager column as NULL).",
     "SELECT e.FirstName || ' ' || e.LastName AS Employee, "
     "m.FirstName || ' ' || m.LastName AS Manager "
     "FROM Employee e LEFT JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "ORDER BY e.EmployeeId"),

    # 6. COUNT DISTINCT.
    ("s_006", 1, "count_distinct",
     "How many distinct countries do we have customers in?",
     "SELECT COUNT(DISTINCT Country) AS CountryCount FROM Customer"),

    # 7. UNION across two tables, deduped.
    ("s_007", 2, "union",
     "List every distinct country that appears either as a customer's country "
     "or as an invoice's billing country. Order alphabetically.",
     "SELECT Country FROM Customer "
     "UNION "
     "SELECT BillingCountry FROM Invoice "
     "ORDER BY Country"),

    # 8. Nested aggregation: AVG over a COUNT-grouped subquery.
    ("s_008", 2, "nested_agg",
     "What is the average number of tracks per album, across all albums?",
     "SELECT AVG(TrackCount) AS AvgTracksPerAlbum "
     "FROM (SELECT COUNT(*) AS TrackCount FROM Track GROUP BY AlbumId)"),

    # 9. COALESCE with LEFT JOIN of a derived count table.
    ("s_009", 2, "coalesce",
     "List all employees and the number of customers they support. Show 0 for "
     "employees who support no customers. Order by customer count descending, "
     "then by employee name.",
     "SELECT e.FirstName || ' ' || e.LastName AS EmployeeName, "
     "COALESCE(c.CustomerCount, 0) AS CustomerCount "
     "FROM Employee e LEFT JOIN ("
     "  SELECT SupportRepId, COUNT(*) AS CustomerCount FROM Customer "
     "  GROUP BY SupportRepId"
     ") c ON e.EmployeeId = c.SupportRepId "
     "ORDER BY CustomerCount DESC, EmployeeName"),

    # 10. LIKE wildcard, case-insensitive.
    ("s_010", 1, "like_wildcard",
     "How many tracks have the word 'love' (case-insensitive) anywhere in their name?",
     "SELECT COUNT(*) AS LoveTrackCount FROM Track WHERE LOWER(Name) LIKE '%love%'"),

    # 11. Correlated subquery: max-per-group via WHERE = (SELECT MAX ...).
    ("s_011", 3, "correlated_subquery",
     "What is the longest track in each genre? Show the genre name, the track "
     "name, and the length in milliseconds. Order by genre name.",
     "SELECT g.Name AS Genre, t.Name AS TrackName, t.Milliseconds "
     "FROM Track t JOIN Genre g ON t.GenreId = g.GenreId "
     "WHERE t.Milliseconds = ("
     "  SELECT MAX(Milliseconds) FROM Track WHERE GenreId = g.GenreId"
     ") ORDER BY g.Name"),

    # 12. CASE used as the GROUP BY key.
    ("s_012", 2, "case_groupby",
     "Bucket invoices by total: small (< $5), medium ($5–$10), large (> $10). "
     "Show the count of invoices in each bucket.",
     "SELECT CASE "
     "  WHEN Total < 5 THEN 'small' "
     "  WHEN Total <= 10 THEN 'medium' "
     "  ELSE 'large' "
     "END AS Bucket, COUNT(*) AS InvoiceCount "
     "FROM Invoice GROUP BY Bucket ORDER BY InvoiceCount DESC"),

    # 13. Date arithmetic: relative window anchored to a subquery'd MAX(date).
    ("s_013", 3, "date_arithmetic",
     "What was the total revenue in the last 6 months of available invoice data?",
     "SELECT SUM(Total) AS Last6MoRevenue FROM Invoice "
     "WHERE InvoiceDate >= date((SELECT MAX(InvoiceDate) FROM Invoice), '-6 months')"),

    # 14. Top-per-group via ROW_NUMBER in a subquery.
    ("s_014", 3, "top_per_group",
     "Who is the top-spending customer for each support rep? Show the rep's "
     "name, the customer's name, and the customer's total spending. Order by "
     "total spending descending.",
     "SELECT EmployeeName, CustomerName, TotalSpent FROM ("
     "  SELECT e.FirstName || ' ' || e.LastName AS EmployeeName, "
     "         c.FirstName || ' ' || c.LastName AS CustomerName, "
     "         SUM(i.Total) AS TotalSpent, "
     "         ROW_NUMBER() OVER (PARTITION BY e.EmployeeId "
     "                            ORDER BY SUM(i.Total) DESC) AS rn "
     "  FROM Employee e "
     "  JOIN Customer c ON e.EmployeeId = c.SupportRepId "
     "  JOIN Invoice i ON c.CustomerId = i.CustomerId "
     "  GROUP BY e.EmployeeId, c.CustomerId, e.FirstName, e.LastName, "
     "           c.FirstName, c.LastName"
     ") WHERE rn = 1 ORDER BY TotalSpent DESC"),

    # 15. LIKE + HAVING together.
    ("s_015", 2, "like_with_having",
     "Which customers in cities starting with 'S' have more than 1 invoice? "
     "Show their name, city, and invoice count. Order by invoice count "
     "descending, then by customer name.",
     "SELECT c.FirstName || ' ' || c.LastName AS CustomerName, c.City, "
     "       COUNT(i.InvoiceId) AS InvoiceCount "
     "FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId "
     "WHERE c.City LIKE 'S%' "
     "GROUP BY c.CustomerId, c.City "
     "HAVING COUNT(i.InvoiceId) > 1 "
     "ORDER BY InvoiceCount DESC, CustomerName"),
]


def main() -> None:
    conn = load_db("data/Chinook.db")
    out = []
    for qid, tier, pattern, question, sql in QUESTIONS:
        rows = query_db(conn, sql, return_as_df=False)
        out.append({
            "id": qid,
            "tier": tier,
            "pattern": pattern,
            "question": question,
            "gold_sql": sql,
            "expected_result": rows,
            "evaluation": "sql_result_match",
        })
        n = len(rows)
        head = rows[0] if rows else None
        print(f"  {qid}  tier={tier}  pattern={pattern:<20s}  rows={n}  first={head}")
    Path("data/synthetic_questions.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote data/synthetic_questions.json with {len(out)} questions")


if __name__ == "__main__":
    main()
