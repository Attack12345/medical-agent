-- 智慧问诊 Agent 数据库 DDL（DEV_DOC §2.3，锁定）
-- 执行：mysql -uroot -p < sql/schema.sql

CREATE DATABASE IF NOT EXISTS medical_agent DEFAULT CHARSET utf8mb4;
USE medical_agent;

-- 用户
CREATE TABLE IF NOT EXISTS user (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(64) NOT NULL UNIQUE,
  password_hash VARCHAR(128) NOT NULL,       -- sha256(salt+password)
  salt VARCHAR(32) NOT NULL,
  role VARCHAR(16) NOT NULL DEFAULT 'USER',  -- USER / ADMIN
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 对话会话
CREATE TABLE IF NOT EXISTS conversation (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id BIGINT NOT NULL,
  title VARCHAR(128) DEFAULT '',
  status VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',  -- ACTIVE / CLOSED
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY idx_user (user_id)
) ENGINE=InnoDB;

-- 消息（用户提问 + Agent 回答，含证据链）
CREATE TABLE IF NOT EXISTS message (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  conversation_id BIGINT NOT NULL,
  role VARCHAR(16) NOT NULL,                 -- USER / ASSISTANT
  content TEXT NOT NULL,
  intent VARCHAR(32),                        -- 意图标签（MEDICAL_QUERY/DEPARTMENT/DRUG/KNOWLEDGE/GUIDE/CHAT）
  evidence_json JSON,                        -- [{type: 'GRAPH_NODE'|'RETRIEVAL', ref: '疾病:高血压', quote: '...'}]
  risk_level VARCHAR(16) DEFAULT 'NONE',     -- NONE/LOW/MEDIUM/HIGH（急症警告）
  disclaimer_added TINYINT(1) DEFAULT 0,     -- 免责声明是否附加（安全层审计）
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_conv (conversation_id)
) ENGINE=InnoDB;

-- 金标评估集
CREATE TABLE IF NOT EXISTS golden_case (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  case_type VARCHAR(16) NOT NULL,            -- INTENT/SAFETY/ANSWER 三类
  question TEXT NOT NULL,
  expected_intent VARCHAR(32),
  expected_risk_level VARCHAR(16) DEFAULT 'NONE',
  expected_disclaimer TINYINT(1) DEFAULT 1,  -- 医疗类是否必须带免责声明
  expected_refusal TINYINT(1) DEFAULT 0,     -- 期望拒答（M7 补）
  expected_entities JSON,                    -- 期望命中的实体列表（科室/疾病/药物）
  expected_keywords JSON,                    -- 回答必须包含的关键词（如'立即就医'）
  remark VARCHAR(512),
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 评估运行
CREATE TABLE IF NOT EXISTS eval_run (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  version VARCHAR(32) NOT NULL,
  metrics_json JSON NOT NULL,
  passed TINYINT(1) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;
