-- Q1. What is total monthly revenue for each year, and what is the Month-over-Month (MoM) and
-- Year-over-Year (YoY) % growth?

-- customers = customer_id,customer_name,segment,region,join_date,email
-- products = product_id,product_name,category,unit_price,margin_pct
-- sales_tansactions = transaction_id,txn_date,product_id,customer_id,region,quantity,unit_price,discount,revenue
-- targets = year,month,target_revenue


select year,month,total_sales,round(((total_sales-prev_month_sales)/prev_month_sales)*100,2) as `MOM%`,
same_month_last_year, round(((total_sales-same_month_last_year)/same_month_last_year)*100,2) as `YOY%`
from (
select year,month,total_sales,lag(total_sales,1) over ( order by year,month) as prev_month_sales,
lag(total_sales,12) over (order by year,month) as same_month_last_year
from(
select year(txn_date) as year, month(txn_date) as month, sum(revenue) as total_sales 
from sales_transactions
group by year,month)dt)dt2
order by year,month;

-- Q2. Which months consistently outperform others across all 3 years (seasonality check)?
SELECT
    MONTH(txn_date) AS mo,
    ROUND(AVG(monthly_total), 2) AS avg_revenue_across_years
FROM (
    SELECT YEAR(txn_date) AS yr, MONTH(txn_date) AS mo2, txn_date,
           SUM(revenue) OVER (PARTITION BY YEAR(txn_date), MONTH(txn_date)) AS monthly_total
    FROM sales_transactions
) t
GROUP BY MONTH(txn_date)
ORDER BY avg_revenue_across_years DESC;


-- Q3. Compound monthly growth rate (CMGR) over the last 12 months
WITH monthly_rev AS (
    SELECT YEAR(txn_date) AS yr, MONTH(txn_date) AS mo, SUM(revenue) AS total_revenue
    FROM sales_transactions
    GROUP BY YEAR(txn_date), MONTH(txn_date)
    ORDER BY yr, mo
    LIMIT 12  
)
SELECT
    MIN(total_revenue) AS start_revenue,
    MAX(total_revenue) AS end_revenue,
    ROUND((POWER(MAX(total_revenue) / NULLIF(MIN(total_revenue),0), 1.0/12) - 1) * 100, 2) AS cmgr_pct
FROM monthly_rev;


-- Q4. Sharpest revenue decline by quarter x region
WITH quarterly AS (
    SELECT
        YEAR(txn_date) AS yr,
        QUARTER(txn_date) AS qtr,
        region,
        SUM(revenue) AS total_revenue
    FROM sales_transactions
    GROUP BY YEAR(txn_date), QUARTER(txn_date), region
)
SELECT
    yr, qtr, region, total_revenue,
    LAG(total_revenue) OVER (PARTITION BY region ORDER BY yr, qtr) AS prev_qtr_revenue,
    ROUND((total_revenue - LAG(total_revenue) OVER (PARTITION BY region ORDER BY yr, qtr))
          / LAG(total_revenue) OVER (PARTITION BY region ORDER BY yr, qtr) * 100, 2) AS qoq_growth_pct
FROM quarterly
ORDER BY qoq_growth_pct ASC
LIMIT 10;


-- ------------------------------------------------------------
-- TARGET-GAP QUESTIONS
-- ------------------------------------------------------------

-- Q5. Actual vs target revenue per month, with gap
SELECT
    t.year, t.month, t.target_revenue,
    COALESCE(SUM(s.revenue), 0) AS actual_revenue,
    t.target_revenue - COALESCE(SUM(s.revenue), 0) AS gap_amount,
    ROUND((t.target_revenue - COALESCE(SUM(s.revenue), 0)) / t.target_revenue * 100, 2) AS gap_pct
FROM targets t
LEFT JOIN sales_transactions s
    ON YEAR(s.txn_date) = t.year AND MONTH(s.txn_date) = t.month
GROUP BY t.year, t.month, t.target_revenue
ORDER BY t.year, t.month;


-- Q6. Region contributing the largest share of shortfall to annual target
WITH region_actual AS (
    SELECT region, YEAR(txn_date) AS yr, SUM(revenue) AS actual_revenue
    FROM sales_transactions
    GROUP BY region, YEAR(txn_date)
),
annual_target AS (
    SELECT year, SUM(target_revenue) AS annual_target
    FROM targets
    GROUP BY year
),
region_share AS (
    SELECT region, yr,
           actual_revenue / SUM(actual_revenue) OVER (PARTITION BY yr) AS revenue_share
    FROM region_actual
)
SELECT
    ra.region, ra.yr, ra.actual_revenue, at.annual_target,
    ROUND(rs.revenue_share * at.annual_target, 2) AS pro_rata_target,
    ROUND(ra.actual_revenue - (rs.revenue_share * at.annual_target), 2) AS shortfall_vs_prorata
FROM region_actual ra
JOIN annual_target at ON ra.yr = at.year
JOIN region_share rs ON ra.region = rs.region AND ra.yr = rs.yr
ORDER BY shortfall_vs_prorata ASC;


-- Q7. Product category furthest behind its pro-rata share of annual target
WITH category_actual AS (
    SELECT p.category, YEAR(s.txn_date) AS yr, SUM(s.revenue) AS actual_revenue
    FROM sales_transactions s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.category, YEAR(s.txn_date)
),
annual_target AS (
    SELECT year, SUM(target_revenue) AS annual_target
    FROM targets
    GROUP BY year
),
category_share AS (
    SELECT category, yr,
           actual_revenue / SUM(actual_revenue) OVER (PARTITION BY yr) AS revenue_share
    FROM category_actual
)
SELECT
    ca.category, ca.yr, ca.actual_revenue,
    ROUND(cs.revenue_share * at.annual_target, 2) AS pro_rata_target,
    ROUND(ca.actual_revenue - (cs.revenue_share * at.annual_target), 2) AS shortfall_vs_prorata
FROM category_actual ca
JOIN annual_target at ON ca.yr = at.year
JOIN category_share cs ON ca.category = cs.category AND ca.yr = cs.yr
ORDER BY shortfall_vs_prorata ASC;


-- Q8. "Business as usual" projection: would last year's avg growth rate hit target?
WITH monthly_rev AS (
    SELECT YEAR(txn_date) AS yr, MONTH(txn_date) AS mo, SUM(revenue) AS total_revenue
    FROM sales_transactions
    GROUP BY YEAR(txn_date), MONTH(txn_date)
),
growth_rates AS (
    SELECT yr, mo, total_revenue,
           (total_revenue - LAG(total_revenue) OVER (ORDER BY yr, mo))
                / LAG(total_revenue) OVER (ORDER BY yr, mo) AS mom_growth
    FROM monthly_rev
)
SELECT ROUND(AVG(mom_growth) * 100, 2) AS avg_mom_growth_pct_last_year
FROM growth_rates
WHERE yr = (SELECT MAX(YEAR(txn_date)) FROM sales_transactions);
-- Use avg_mom_growth_pct_last_year to project remaining months in Python/Excel
-- and compare the projected total against targets.target_revenue.


-- ------------------------------------------------------------
-- SECTION 1.3 — PRODUCT & CATEGORY QUESTIONS
-- ------------------------------------------------------------

-- Q9. Products/categories with declining 3-month rolling average trend
with monthly_sales as 
(
SELECT product_id, YEAR(txn_date) AS yr, MONTH(txn_date) AS mo, SUM(revenue) AS monthly_revenue
FROM sales_transactions
GROUP BY product_id, YEAR(txn_date), MONTH(txn_date)),

rolling_sales as 
( select product_id, yr,mo,monthly_revenue, 
round(avg(monthly_revenue) over ( partition by product_id order by yr,mo 
			rows between 2 preceding and current row),2) as rolling_average_3mo
from monthly_sales),

trend as 
( select * ,LAG(rolling_average_3mo,2) over (partition by product_id order by yr,mo) as previous_rolling_average
from rolling_sales )

select product_id,yr,mo,monthly_revenue,rolling_average_3mo,previous_rolling_average,
case 
	when rolling_average_3mo < previous_rolling_average
    then "Declining"
    else "Not Declining"
end as trend
from trend
order by product_id,yr,mo;
	

-- Q10. Revenue concentration by product
WITH product_revenue AS (
    SELECT product_id, SUM(revenue) AS total_revenue
    FROM sales_transactions
    GROUP BY product_id
),
ranked AS (
    SELECT product_id, total_revenue,
           SUM(total_revenue) OVER (ORDER BY total_revenue DESC) AS running_total,
           SUM(total_revenue) OVER () AS grand_total
    FROM product_revenue
)
SELECT
    product_id, total_revenue,
    ROUND(running_total / grand_total * 100, 2) AS cumulative_pct
FROM ranked
ORDER BY total_revenue DESC;


-- Q11. Discount level vs volume vs net revenue, by category
SELECT
    p.category,
    ROUND(AVG(s.discount), 3) AS avg_discount,
    SUM(s.quantity) AS total_quantity,
    ROUND(SUM(s.revenue), 2) AS total_revenue
FROM sales_transactions s
JOIN products p 
ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY avg_discount DESC;


-- ------------------------------------------------------------
-- SECTION 1.4 — CUSTOMER & SEGMENTATION QUESTIONS
-- ------------------------------------------------------------

-- Q12. Customer-level RFM base (last 12 months)
SELECT
    customer_id,
    DATEDIFF(CURDATE(), MAX(txn_date)) AS recency_days,
    COUNT(*) AS frequency_orders,
    ROUND(SUM(revenue), 2) AS monetary_total,
    ROUND(AVG(revenue), 2) AS avg_order_value
FROM sales_transactions
WHERE txn_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY customer_id
ORDER BY monetary_total DESC;


-- Q13. Revenue per customer and YoY growth, by segment
WITH seg_yearly AS (
    SELECT c.segment, YEAR(s.txn_date) AS yr,
           SUM(s.revenue) AS total_revenue,
           COUNT(DISTINCT s.customer_id) AS customer_count
    FROM sales_transactions s
    JOIN customers c ON s.customer_id = c.customer_id
    GROUP BY c.segment, YEAR(s.txn_date)
)
SELECT
    segment, yr, total_revenue, customer_count,
    ROUND(total_revenue / customer_count, 2) AS revenue_per_customer,
    ROUND((total_revenue - LAG(total_revenue) OVER (PARTITION BY segment ORDER BY yr))
          / LAG(total_revenue) OVER (PARTITION BY segment ORDER BY yr) * 100, 2) AS yoy_growth_pct
FROM seg_yearly
ORDER BY segment, yr;


-- Q14. Customers active last year but silent in the last 3/6 months (churn risk)
SELECT DISTINCT c.customer_id, c.customer_name, c.segment, MAX(s2.txn_date) AS last_purchase
FROM customers c
JOIN sales_transactions s1
    ON c.customer_id = s1.customer_id
    AND s1.txn_date >= DATE_SUB(CURDATE(), INTERVAL 15 MONTH)
    AND s1.txn_date <  DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
LEFT JOIN sales_transactions s2
    ON c.customer_id = s2.customer_id
    AND s2.txn_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
GROUP BY c.customer_id, c.customer_name, c.segment
HAVING last_purchase IS NULL;


-- Q15. Repeat-purchase rate by region and segment
WITH customer_months AS (
    SELECT customer_id, region,
           COUNT(DISTINCT DATE_FORMAT(txn_date, '%Y-%m')) AS active_months
    FROM sales_transactions
    GROUP BY customer_id, region
)
SELECT
    region,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN active_months > 1 THEN 1 ELSE 0 END) AS repeat_customers,
    ROUND(SUM(CASE WHEN active_months > 1 THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS repeat_rate_pct
FROM customer_months
GROUP BY region
ORDER BY repeat_rate_pct DESC;


-- ------------------------------------------------------------
-- SECTION 1.5 — REGIONAL QUESTIONS
-- ------------------------------------------------------------

-- Q16. Fastest growing / shrinking region YoY
WITH region_yearly AS (
    SELECT region, YEAR(txn_date) AS yr, SUM(revenue) AS total_revenue
    FROM sales_transactions
    GROUP BY region, YEAR(txn_date)
)
SELECT
    region, yr, total_revenue,
    LAG(total_revenue) OVER (PARTITION BY region ORDER BY yr) AS prev_year_revenue,
    ROUND((total_revenue - LAG(total_revenue) OVER (PARTITION BY region ORDER BY yr))
          / LAG(total_revenue) OVER (PARTITION BY region ORDER BY yr) * 100, 2) AS yoy_growth_pct
FROM region_yearly
ORDER BY yoy_growth_pct DESC;


-- Q17. Revenue decomposition by region: customers x orders/customer x avg order value
WITH region_yearly AS (
    SELECT
        region,
        YEAR(txn_date) AS yr,
        COUNT(DISTINCT customer_id) AS active_customers,
        COUNT(*) AS total_orders,
        SUM(revenue) AS total_revenue
    FROM sales_transactions
    GROUP BY region, YEAR(txn_date)
)
SELECT
    region, yr, active_customers, total_orders, total_revenue,
    ROUND(total_orders / active_customers, 2) AS orders_per_customer,
    ROUND(total_revenue / total_orders, 2) AS avg_order_value,
    LAG(active_customers) OVER (PARTITION BY region ORDER BY yr)      AS prev_customers,
    LAG(total_orders / active_customers) OVER (PARTITION BY region ORDER BY yr) AS prev_orders_per_customer,
    LAG(total_revenue / total_orders) OVER (PARTITION BY region ORDER BY yr)    AS prev_avg_order_value
FROM region_yearly
ORDER BY region, yr;





