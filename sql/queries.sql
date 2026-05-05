-- Load raw sales rows with a computed gross_sales column
SELECT
    order_id,
    order_date,
    ship_date,
    region,
    state,
    city,
    sales_channel,
    customer_segment,
    category,
    sub_category,
    product_name,
    units_sold,
    unit_price,
    unit_cost,
    discount_pct,
    marketing_spend,
    inventory_days,
    returned,
    customer_rating,
    units_sold * unit_price AS gross_sales
FROM retail_sales;

-- Example aggregation query
SELECT
    region,
    sales_channel,
    COUNT(*) AS orders,
    SUM(units_sold) AS total_units,
    ROUND(SUM(units_sold * unit_price * (1 - discount_pct)), 2) AS net_revenue
FROM retail_sales
GROUP BY region, sales_channel
ORDER BY net_revenue DESC;
