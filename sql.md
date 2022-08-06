# sql学习笔记

## count(1)与count(*)比较

如果表**没有主键**，那么count(1)比count(*)快。
如果表**有主键**，那么count(key value, union key value)比count(*)快。
如果表**只有一个字段**，count(*)最快。
