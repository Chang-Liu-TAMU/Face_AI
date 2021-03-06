CREATE TABLE IF NOT EXISTS `sys_user` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `user_id` VARCHAR(32) NOT NULL UNIQUE,
    `nickname` VARCHAR(128),
    `username` VARCHAR(128) NOT NULL UNIQUE,
    `hashed_password` VARCHAR(128) NOT NULL,
    `authority_id` INT NOT NULL  DEFAULT 100,
    `is_active` INT NOT NULL  DEFAULT 0,
    `is_delete` INT NOT NULL  DEFAULT 0,
    `create_time` DATETIME(6)   DEFAULT CURRENT_TIMESTAMP(6),
    `update_time` DATETIME(6)   DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COMMENT='save sys user info';