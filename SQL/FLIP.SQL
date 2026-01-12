-- ==========================================
-- 1️⃣ Add Foreign Key Constraints
-- ==========================================
ALTER TABLE dbo.orders_data
ADD CONSTRAINT fk_orders_data_customer_id
FOREIGN KEY (customer_id) REFERENCES dbo.customers_data(customer_id);

ALTER TABLE dbo.order_items_data
ADD CONSTRAINT fk_order_items_order_id
FOREIGN KEY (order_id) REFERENCES dbo.orders_data(order_id);

ALTER TABLE dbo.order_items_data
ADD CONSTRAINT fk_order_items_product_id
FOREIGN KEY (product_id) REFERENCES dbo.product_data(product_id);

ALTER TABLE dbo.delivery_data
ADD CONSTRAINT fk_delivery_order_id
FOREIGN KEY (order_id) REFERENCES dbo.orders_data(order_id);

ALTER TABLE dbo.returns_data
ADD CONSTRAINT fk_returns_order_item_id
FOREIGN KEY (order_item_id) REFERENCES dbo.order_items_data(order_item_id);

-- ==========================================
-- 2️⃣ Select all joined data
-- ==========================================
SELECT * 
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id;

-- ==========================================
-- 3️⃣ Gross Revenue vs Net Revenue
-- ==========================================
SELECT 
    SUM(ot.quantity * ot.selling_price) AS gross_revenue,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS net_revenue
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id;

-- ==========================================
-- 4️⃣ Delayed vs On-time Orders
-- ==========================================
SELECT 
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT CASE WHEN d.delivery_status = 'Delayed' THEN o.order_id END) AS delayed_orders,
    COUNT(DISTINCT CASE WHEN d.delivery_status <> 'Delayed' THEN o.order_id END) AS on_time_orders
FROM dbo.orders_data o
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id;

-- ==========================================
-- 5️⃣ Overall Sales and Returns
-- ==========================================
SELECT
    SUM(ot.quantity * ot.selling_price) AS gross_revenue,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS net_revenue,
    SUM((ot.quantity * ot.selling_price) - CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS leakage,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_returns,
    SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_orders,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS return_percentage
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id;

-- ==========================================
-- 6️⃣ Age Group-wise Revenue, Returns, Leakage
-- ==========================================
SELECT 
    c.age_group,
    SUM(ot.quantity * ot.selling_price) AS gross_revenue,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS net_revenue,
    SUM((ot.quantity * ot.selling_price) - CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS leakage,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_returns,
    SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_orders,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS return_percentage
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id
GROUP BY c.age_group;

-- ==========================================
-- 7️⃣ Product-wise Revenue, Returns, Leakage
-- ==========================================
SELECT 
    p.product_id,
    SUM(ot.quantity * ot.selling_price) AS gross_revenue,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS net_revenue,
    SUM((ot.quantity * ot.selling_price) - CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS leakage,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_returns,
    SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_orders,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS return_percentage
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id
GROUP BY p.product_id;

-- ==========================================
-- 8️⃣ Product-wise Delivery Performance
-- ==========================================
SELECT
    p.product_id,
    COUNT(ot.order_item_id) AS total_orders,
    SUM(CASE WHEN d.delivery_status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_orders,
    SUM(CASE WHEN d.delivery_status = 'On-time' THEN 1 ELSE 0 END) AS on_time_orders,
    SUM(CASE WHEN d.delivery_status = 'Delayed' THEN 1 ELSE 0 END) * 100.0 / COUNT(ot.order_item_id) AS delayed_percentage
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
GROUP BY p.product_id;

-- ==========================================
-- 9️⃣ Top 10 Customers by Revenue
-- ==========================================
SELECT TOP 10
    c.customer_id,
    SUM(ot.quantity * ot.selling_price) AS gross_revenue,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS net_revenue,
    SUM((ot.quantity * ot.selling_price) - CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS leakage,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_returns,
    SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_orders,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS return_percentage
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id
GROUP BY c.customer_id
ORDER BY net_revenue DESC;

-- ==========================================
-- 10️⃣ Monthly Gross vs Net Revenue Trend
-- ==========================================
SELECT
    DATENAME(MONTH, o.order_date) AS month_name,
    MONTH(o.order_date) AS month_num,
    SUM(ot.quantity * ot.selling_price) AS gross_revenue,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS net_revenue
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id
GROUP BY DATENAME(MONTH, o.order_date), MONTH(o.order_date)
ORDER BY month_num ASC;

-- ==========================================
-- 11️⃣ Yearly Gross vs Net Revenue
-- ==========================================
SELECT
    YEAR(o.order_date) AS order_year,
    SUM(ot.quantity * ot.selling_price) AS gross_revenue,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS net_revenue
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id
GROUP BY YEAR(o.order_date)
ORDER BY order_year ASC;

-- ==========================================
-- 12️⃣ Total Customers and Products
-- ==========================================
SELECT COUNT(DISTINCT c.customer_id) AS total_customers
FROM dbo.customers_data c;

SELECT COUNT(p.product_id) AS total_products
FROM dbo.product_data p;

-- ==========================================
-- 13️⃣ Orders without Returns
-- ==========================================
SELECT COUNT(DISTINCT o.order_id) AS total_orders_without_returns
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id
WHERE r.order_item_id IS NULL;

-- ==========================================
-- 14️⃣ Top 10 Categories by Revenue
-- ==========================================
SELECT TOP 10
    p.category,
    SUM(ot.quantity * ot.selling_price) AS gross_revenue,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS net_revenue,
    SUM((ot.quantity * ot.selling_price) - CASE WHEN r.order_item_id IS NOT NULL THEN 0 ELSE ot.quantity * ot.selling_price END) AS leakage,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_returns,
    SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS total_orders,
    SUM(CASE WHEN r.order_item_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN ot.order_item_id IS NOT NULL THEN 1 ELSE 0 END) AS return_percentage
FROM dbo.customers_data c
INNER JOIN dbo.orders_data o ON o.customer_id = c.customer_id
INNER JOIN dbo.order_items_data ot ON ot.order_id = o.order_id
INNER JOIN dbo.product_data p ON p.product_id = ot.product_id
INNER JOIN dbo.delivery_data d ON d.order_id = o.order_id
LEFT JOIN dbo.returns_data r ON r.order_item_id = ot.order_item_id
GROUP BY p.category
ORDER BY net_revenue DESC;
