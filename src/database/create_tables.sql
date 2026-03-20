CREATE TABLE IF NOT EXISTS subjects
(
    id UInt64,
    pid UInt64,
    bin UInt64 DEFAULT 0,
    iin UInt64 DEFAULT 0,
    name_ru String,
    name_kz String,
    is_customer UInt8 DEFAULT 0,
    is_organizer UInt8 DEFAULT 0,
    is_supplier UInt8 DEFAULT 0,
    register_date DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now(),

    INDEX idx_subjects_bin bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_subjects_register_date register_date TYPE minmax GRANULARITY 1,
    INDEX idx_subjects_updated_at updated_at TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(updated_at)
ORDER BY (bin, id, updated_at);


CREATE TABLE IF NOT EXISTS plans
(
    id UInt64,
    plan_number String,
    customer_bin UInt64,
    organizer_bin UInt64 DEFAULT 0,
    enstr_code String DEFAULT '',
    enstr_code_lvl2 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 2),
    enstr_code_lvl3 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 3),
    enstr_code_lvl4 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 4),
    kato_code String DEFAULT '',
    mkei_code String DEFAULT '',
    planned_amount Decimal128(2) DEFAULT 0,
    quantity Decimal128(3) DEFAULT 0,
    ref_plan_status_id UInt64 DEFAULT 0,
    plan_year UInt16 DEFAULT 0,
    publish_date DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now(),

    INDEX idx_plans_customer_bin customer_bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_plans_enstr_code enstr_code TYPE set(2048) GRANULARITY 2,
    INDEX idx_plans_kato_code kato_code TYPE set(2048) GRANULARITY 2,
    INDEX idx_plans_publish_date publish_date TYPE minmax GRANULARITY 1,
    INDEX idx_plans_updated_at updated_at TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(publish_date)
ORDER BY (customer_bin, publish_date, id);


CREATE TABLE IF NOT EXISTS announcements
(
    id UInt64,
    number_anno String,
    customer_bin UInt64,
    organizer_bin UInt64 DEFAULT 0,
    total_sum Decimal128(2) DEFAULT 0,
    count_lots UInt32 DEFAULT 0,
    ref_trade_methods_id UInt64 DEFAULT 0,
    ref_buy_status_id UInt64 DEFAULT 0,
    publish_date DateTime DEFAULT now(),
    start_date DateTime DEFAULT now(),
    end_date DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now(),

    INDEX idx_ann_customer_bin customer_bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_ann_publish_date publish_date TYPE minmax GRANULARITY 1,
    INDEX idx_ann_start_date start_date TYPE minmax GRANULARITY 1,
    INDEX idx_ann_end_date end_date TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(publish_date)
ORDER BY (customer_bin, publish_date, id);


CREATE TABLE IF NOT EXISTS lots
(
    id UInt64,
    lot_number String,
    trd_buy_id UInt64,
    customer_bin UInt64,
    supplier_bin UInt64 DEFAULT 0,
    enstr_code String DEFAULT '',
    enstr_code_lvl2 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 2),
    enstr_code_lvl3 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 3),
    enstr_code_lvl4 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 4),
    kato_code String DEFAULT '',
    mkei_code String DEFAULT '',
    quantity Decimal128(3) DEFAULT 0,
    amount Decimal128(2) DEFAULT 0,
    unit_price Decimal128(2) DEFAULT 0,
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now(),

    INDEX idx_lots_customer_bin customer_bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_lots_supplier_bin supplier_bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_lots_enstr_code enstr_code TYPE set(2048) GRANULARITY 2,
    INDEX idx_lots_kato_code kato_code TYPE set(2048) GRANULARITY 2,
    INDEX idx_lots_created_at created_at TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(created_at)
ORDER BY (trd_buy_id, enstr_code_lvl4, customer_bin, id);


CREATE TABLE IF NOT EXISTS contracts
(
    id UInt64,
    contract_number String,
    trd_buy_id UInt64 DEFAULT 0,
    lot_id UInt64 DEFAULT 0,
    customer_bin UInt64,
    supplier_bin UInt64,
    enstr_code String DEFAULT '',
    enstr_code_lvl2 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 2),
    enstr_code_lvl3 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 3),
    enstr_code_lvl4 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 4),
    kato_code String DEFAULT '',
    contract_sum Decimal128(2) DEFAULT 0,
    contract_sum_wnds Decimal128(2) DEFAULT 0,
    ref_contract_status_id UInt64 DEFAULT 0,
    sign_date DateTime DEFAULT now(),
    start_date DateTime DEFAULT now(),
    end_date DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now(),

    INDEX idx_contracts_customer_bin customer_bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_contracts_supplier_bin supplier_bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_contracts_enstr_code enstr_code TYPE set(2048) GRANULARITY 2,
    INDEX idx_contracts_kato_code kato_code TYPE set(2048) GRANULARITY 2,
    INDEX idx_contracts_sign_date sign_date TYPE minmax GRANULARITY 1,
    INDEX idx_contracts_start_date start_date TYPE minmax GRANULARITY 1,
    INDEX idx_contracts_end_date end_date TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(sign_date)
ORDER BY (supplier_bin, customer_bin, sign_date, id);


CREATE TABLE IF NOT EXISTS contract_acts
(
    id UInt64,
    contract_id UInt64,
    act_number String,
    customer_bin UInt64 DEFAULT 0,
    supplier_bin UInt64 DEFAULT 0,
    act_sum Decimal128(2) DEFAULT 0,
    act_date DateTime DEFAULT now(),
    approve_date DateTime DEFAULT now(),
    ref_act_status_id UInt64 DEFAULT 0,
    updated_at DateTime DEFAULT now(),

    INDEX idx_acts_contract_id contract_id TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_acts_customer_bin customer_bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_acts_supplier_bin supplier_bin TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_acts_act_date act_date TYPE minmax GRANULARITY 1,
    INDEX idx_acts_approve_date approve_date TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(act_date)
ORDER BY (contract_id, act_date, id);


CREATE TABLE IF NOT EXISTS reference_enstr
(
    enstr_code String,
    enstr_code_lvl2 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 2),
    enstr_code_lvl3 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 3),
    enstr_code_lvl4 String MATERIALIZED substring(replaceRegexpAll(enstr_code, '[^0-9]', ''), 1, 4),
    name_ru String,
    name_kz String,
    level UInt8 DEFAULT 0,
    parent_code String DEFAULT '',
    is_active UInt8 DEFAULT 1,
    updated_at DateTime DEFAULT now(),

    INDEX idx_ref_enstr_code enstr_code TYPE set(4096) GRANULARITY 1,
    INDEX idx_ref_enstr_updated_at updated_at TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(updated_at)
ORDER BY (enstr_code, updated_at);


CREATE TABLE IF NOT EXISTS reference_kato
(
    kato_code String,
    name_ru String,
    name_kz String,
    level UInt8 DEFAULT 0,
    parent_code String DEFAULT '',
    is_active UInt8 DEFAULT 1,
    updated_at DateTime DEFAULT now(),

    INDEX idx_ref_kato_code kato_code TYPE set(4096) GRANULARITY 1,
    INDEX idx_ref_kato_updated_at updated_at TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(updated_at)
ORDER BY (kato_code, updated_at);


CREATE TABLE IF NOT EXISTS reference_mkei
(
    mkei_code String,
    name_ru String,
    name_kz String,
    short_name String DEFAULT '',
    is_active UInt8 DEFAULT 1,
    updated_at DateTime DEFAULT now(),

    INDEX idx_ref_mkei_code mkei_code TYPE set(4096) GRANULARITY 1,
    INDEX idx_ref_mkei_updated_at updated_at TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(updated_at)
ORDER BY (mkei_code, updated_at);


CREATE TABLE IF NOT EXISTS anomaly_results
(
    id UInt64,
    anomaly_type String,
    entity_type String,
    entity_id UInt64,
    detected_at DateTime,
    severity String,
    deviation_pct Decimal128(2),
    expected_value Decimal128(2),
    actual_value Decimal128(2),
    sample_size UInt32,
    enstr_code String,
    kato_code Nullable(String),
    metadata String,
    updated_at DateTime DEFAULT now(),

    INDEX idx_anomaly_results_anomaly_type anomaly_type TYPE set(16) GRANULARITY 1,
    INDEX idx_anomaly_results_entity_type entity_type TYPE set(16) GRANULARITY 1,
    INDEX idx_anomaly_results_entity_id entity_id TYPE bloom_filter(0.01) GRANULARITY 2,
    INDEX idx_anomaly_results_detected_at detected_at TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(detected_at)
ORDER BY (anomaly_type, entity_type, entity_id, detected_at, id);


CREATE MATERIALIZED VIEW IF NOT EXISTS mv_spend_by_bin
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(sign_date)
ORDER BY (customer_bin, sign_date)
AS
SELECT
    customer_bin,
    toDate(sign_date) AS sign_date,
    sum(contract_sum) AS total_contract_sum,
    count() AS contract_count
FROM
(
    SELECT
        id,
        argMax(customer_bin, updated_at) AS customer_bin,
        argMax(contract_sum, updated_at) AS contract_sum,
        argMax(sign_date, updated_at) AS sign_date
    FROM contracts
    GROUP BY id
)
WHERE contract_sum > 0
GROUP BY customer_bin, sign_date;


CREATE MATERIALIZED VIEW IF NOT EXISTS mv_spend_by_enstr
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(sign_date)
ORDER BY (enstr_code_lvl4, sign_date)
AS
SELECT
    enstr_code_lvl4,
    toDate(sign_date) AS sign_date,
    sum(contract_sum) AS total_spend,
    count() AS contract_count
FROM
(
    SELECT
        id,
        argMax(enstr_code_lvl4, updated_at) AS enstr_code_lvl4,
        argMax(contract_sum, updated_at) AS contract_sum,
        argMax(sign_date, updated_at) AS sign_date
    FROM contracts
    GROUP BY id
)
WHERE contract_sum > 0 AND length(enstr_code_lvl4) = 4
GROUP BY enstr_code_lvl4, sign_date;
