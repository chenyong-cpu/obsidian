# Python爬虫教程

## 预备知识

### 爬虫分类

- 通用爬虫：抓取系统重要组成部分，抓取的是一整张页面数据。
- 聚焦爬虫：建立在通用爬虫的基础之上，抓取的是页面中特定的局部内容。
- 增量式爬虫：检测网站中数据更新的情况，只会抓取网站中最新更新出来的数据。

### 反爬机制

- robots.txt协议（君子协议）：规定了网站中哪些数据可以被爬虫爬取，哪些数据不可以被爬取。

### http协议

- 概念：服务器和客户端进行数据交互的一种形式。

### 常用请求头信息

- User-Agent：请求载体的身份标识（通常是某种浏览器）。
- Connection：请求完毕后，是断开连接还是保持连接。

### 常用响应头协议

- Content-Type：服务端响应客户端的数据类型。

### https协议

- 概念：安全的超文本传输协议，对进行数据加密（证书密钥加密）。

### 加密方式

- 对称密钥加密：客户端将密钥和密文一起发送给服务端，服务端进行解密。
- 非对称密钥加密：
	- 服务端创建密匙对
	- 服务端将公匙发送给客户端
	- 客户端使用公匙进行加密
	- 客户端将密文发送给服务端
	- 服务端使用私钥进行解密
- 证书密钥加密
	- 服务端公开密钥提交到证书认证机构
	- 证书认证机构给密钥进行签名

## Requests

Python中原生的一款基于网络请求的模块，功能非常强大，简单便捷，效率极高，可以模拟浏览器发送请求。

### 使用流程

- 指定url

```python
import requests

if __name__ == '__main__':
	url = 'http://125.35.6.84:81'
	params = {
		'xk': '1'
	}
	headers = {
		'User-Agent': '浏览器型号'
	}
	page_text = requests.get(url=url, params=params, headers=headers).text
```

- 发起请求
- 获取响应数据
- 持久化存储

### 动态页面分析

- 检查页面数据是否是动态数据，可以页面查看源代码
- 如果页面数据时动态数据，需要分析浏览器网络请求中‘Fetch/XHR’中的请求
- 根据动态数据进行深入分析，可以分析多个页面之间的关系

## 数据解析

### 正则

- ‘.*’单个字符匹配任意次，即贪婪匹配
- ‘.*?’是满足条件的情况只匹配一次，即最小匹配
- re.S标识单行匹配，通常都是re.S

```python
import re

s1 = '萧炎7药岩6岩枭3佛怒火莲6'  
re.findall('[0-9]',s1)
# output: ['7', '6', '3', '6']

re.findall('[1-6]',s1)
# output: ['6', '3', '6']

s2='xyz,xcz,xfz,xdz,xaz,xez'  
re.findall('x[de]z',s2)
# output: ['xdz', 'xez']
```

### bs4

### xpath
