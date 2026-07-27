-- Sales & Revenue Analytics Dashboard - PostgreSQL Schema
-- Database: sales_analytics

CREATE TABLE IF NOT EXISTS customers (
    customer_id   VARCHAR(20)  PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    product_id        SERIAL       PRIMARY KEY,
    product_name      VARCHAR(100) NOT NULL UNIQUE,
    product_category  VARCHAR(50)  NOT NULL,
    created_at        TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      VARCHAR(20)  PRIMARY KEY,
    order_date    DATE         NOT NULL,
    customer_id   VARCHAR(20)  NOT NULL REFERENCES customers(customer_id),
    region        VARCHAR(50)  NOT NULL,
    sales_channel VARCHAR(50)  NOT NULL,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sales (
    sale_id       SERIAL        PRIMARY KEY,
    order_id      VARCHAR(20)   NOT NULL REFERENCES orders(order_id),
    product_name  VARCHAR(100)  NOT NULL REFERENCES products(product_name),
    quantity      INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price    NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0),
    discount      NUMERIC(5,4)  NOT NULL DEFAULT 0 CHECK (discount >= 0 AND discount <= 1),
    revenue       NUMERIC(12,2) NOT NULL,
    profit        NUMERIC(12,2) NOT NULL,
    created_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_region ON orders(region);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_sales_order ON sales(order_id);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(product_category);

-- Analytical view for dashboard queries
CREATE OR REPLACE VIEW vw_sales_summary AS
SELECT
    o.order_id,
    o.order_date,
    c.customer_id,
    c.customer_name,
    p.product_category,
    p.product_name,
    o.region,
    s.quantity,
    s.unit_price,
    s.discount,
    s.revenue,
    s.profit,
    o.sales_channel
FROM sales s
JOIN orders o ON s.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON s.product_name = p.product_name;
