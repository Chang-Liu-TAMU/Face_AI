CREATE TABLE IF NOT EXISTS `sys_api` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `path` VARCHAR(128),
    `description` VARCHAR(64),
    `api_group` VARCHAR(32),
    `method` VARCHAR(16),
    `is_delete` INT NOT NULL  DEFAULT 0,
    `create_time` DATETIME(6)   DEFAULT CURRENT_TIMESTAMP(6),
    `update_time` DATETIME(6)   DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COMMENT='save sys api info';
