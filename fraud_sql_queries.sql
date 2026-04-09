SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) AS legitimate_transactions,
    ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_percentage,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS total_fraud_amount,
    ROUND(SUM(CASE WHEN Class = 0 THEN Amount ELSE 0 END), 2) AS total_legitimate_amount,
    ROUND(AVG(CASE WHEN Class = 1 THEN Amount END), 2) AS avg_fraud_amount,
    ROUND(AVG(CASE WHEN Class = 0 THEN Amount END), 2) AS avg_legitimate_amount
FROM transactions;



SELECT 
    CASE 
        WHEN Amount < 50 THEN '1. Under $50'
        WHEN Amount < 100 THEN '2. $50-$100'
        WHEN Amount < 250 THEN '3. $100-$250'
        WHEN Amount < 500 THEN '4. $250-$500'
        WHEN Amount < 1000 THEN '5. $500-$1000'
        ELSE '6. Over $1000'
    END AS amount_bucket,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS fraud_loss_amount
FROM transactions
GROUP BY amount_bucket
ORDER BY amount_bucket;


SELECT 
    CASE 
        WHEN (Time % 86400) / 3600 BETWEEN 0 AND 5 THEN '1. Night (12AM-6AM)'
        WHEN (Time % 86400) / 3600 BETWEEN 6 AND 11 THEN '2. Morning (6AM-12PM)'
        WHEN (Time % 86400) / 3600 BETWEEN 12 AND 17 THEN '3. Afternoon (12PM-6PM)'
        ELSE '4. Evening (6PM-12AM)'
    END AS time_period,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate,
    ROUND(AVG(Amount), 2) AS avg_transaction_amount
FROM transactions
GROUP BY time_period
ORDER BY time_period;


SELECT 
    Amount,
    Time,
    ROUND((Time % 86400) / 3600, 1) AS hour_of_day,
    V1, V2, V3, V4,  -- Top PCA features correlated with fraud
    Class,
    CASE 
        WHEN Class = 1 THEN 'FRAUD' 
        ELSE 'Legitimate' 
    END AS transaction_type
FROM transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 20;


WITH amount_stats AS (
    SELECT 
        Class,
        CASE WHEN Class = 1 THEN 'Fraud' ELSE 'Legitimate' END AS type,
        COUNT(*) AS transaction_count,
        ROUND(MIN(Amount), 2) AS min_amount,
        ROUND(MAX(Amount), 2) AS max_amount,
        ROUND(AVG(Amount), 2) AS mean_amount
    FROM transactions
    GROUP BY Class
)
SELECT * FROM amount_stats;


SELECT 
    CAST((Time % 86400) / 3600 AS INTEGER) AS hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate,
    ROUND(SUM(Amount), 2) AS total_transaction_volume
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;



SELECT 
    'High Value Fraud (>$500)' AS category,
    COUNT(*) AS fraud_count,
    ROUND(SUM(Amount), 2) AS total_loss,
    ROUND(AVG(Amount), 2) AS avg_loss,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions WHERE Class = 1), 2) AS pct_of_all_fraud
FROM transactions
WHERE Class = 1 AND Amount > 500

UNION ALL

SELECT 
    'Medium Value Fraud ($100-$500)' AS category,
    COUNT(*) AS fraud_count,
    ROUND(SUM(Amount), 2) AS total_loss,
    ROUND(AVG(Amount), 2) AS avg_loss,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions WHERE Class = 1), 2) AS pct_of_all_fraud
FROM transactions
WHERE Class = 1 AND Amount BETWEEN 100 AND 500

UNION ALL

SELECT 
    'Low Value Fraud (<$100)' AS category,
    COUNT(*) AS fraud_count,
    ROUND(SUM(Amount), 2) AS total_loss,
    ROUND(AVG(Amount), 2) AS avg_loss,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions WHERE Class = 1), 2) AS pct_of_all_fraud
FROM transactions
WHERE Class = 1 AND Amount < 100;


WITH time_diffs AS (
    SELECT 
        Time,
        Amount,
        Class,
        LAG(Time) OVER (ORDER BY Time) AS prev_time,
        Time - LAG(Time) OVER (ORDER BY Time) AS time_since_last
    FROM transactions
    WHERE Class = 1
)
SELECT 
    CASE 
        WHEN time_since_last < 60 THEN 'Within 1 minute'
        WHEN time_since_last < 300 THEN '1-5 minutes'
        WHEN time_since_last < 900 THEN '5-15 minutes'
        WHEN time_since_last < 3600 THEN '15-60 minutes'
        ELSE 'Over 1 hour'
    END AS time_gap,
    COUNT(*) AS fraud_count,
    ROUND(AVG(Amount), 2) AS avg_amount
FROM time_diffs
WHERE time_since_last IS NOT NULL
GROUP BY time_gap
ORDER BY 
    CASE time_gap
        WHEN 'Within 1 minute' THEN 1
        WHEN '1-5 minutes' THEN 2
        WHEN '5-15 minutes' THEN 3
        WHEN '15-60 minutes' THEN 4
        ELSE 5
    END;
