create database ai_sales_analysis;
use ai_sales_analysis;

-- CUSTOMERS

CREATE TABLE customers (
    customer_id     VARCHAR(10)  PRIMARY KEY,
    customer_name   VARCHAR(100) NOT NULL,
    segment         VARCHAR(20) ,
    region          VARCHAR(20) ,
    join_date       DATE,
    email           VARCHAR(100)
);

-- PRODUCTS

CREATE TABLE products (
    product_id      VARCHAR(10)   PRIMARY KEY,
    product_name    VARCHAR(150)  NOT NULL,
    category        VARCHAR(30) ,
    unit_price      NUMERIC(10,2) ,
    margin_pct      NUMERIC(4,2) 
);

-- SALES_TRANSACTIONS

CREATE TABLE sales_transactions (
    transaction_id  VARCHAR(12)   PRIMARY KEY,
    txn_date        DATE          NOT NULL,
    product_id      VARCHAR(10)   NOT NULL REFERENCES products(product_id),
    customer_id     VARCHAR(10)   NOT NULL REFERENCES customers(customer_id),
    region          VARCHAR(20),
    quantity        INTEGER       CHECK (quantity > 0),
    unit_price      NUMERIC(10,2) CHECK (unit_price >= 0),
    discount        NUMERIC(4,2) ,
    revenue         NUMERIC(12,2) 
);

-- TARGETS  (business goals, monthly grain)

CREATE TABLE targets (
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    target_revenue  NUMERIC(14,2) NOT NULL,
    PRIMARY KEY (year, month)
);


-- AI_RECOMMENDATIONS  

CREATE TABLE ai_recommendations (
    recommendation_id  SERIAL PRIMARY KEY,       
    date_generated      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_amount        NUMERIC(14,2) NOT NULL,
    period_year           INTEGER NOT NULL,
    gap_amount            NUMERIC(14,2),
    gap_pct                 NUMERIC(5,2),
    summary_text          TEXT,
    top_actions            TEXT,                   
    target_segments      TEXT                     
);

















