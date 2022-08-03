# Speech and Language processing

## Introduction

## 1. Regular Expressions, Text Normalization, Edit Distance（正则表达式、文本规范化、编辑距离）

### 1.1 Regular Expressions

#### 1.1.1 基础正则表达式模式

最简单的正则表达式是字符匹配，即把字符按顺序排序。
正则表达式是大小写敏感的，可以使用[wW]来匹配w或者W。
使用‘-’来选择一个范围，即通过[1-9]替代[123456789]。
使用‘^’表示not，通常在[]中，其中[\^wW]表示不为[wW]的字符序列。

使用‘?’在字符后表示该字符出现0次或者1次。
使用‘\*’在字符后表示该字符出现0次及以上。
使用‘+’在字符后表示该字符出现1次及以上。
使用‘.’可以表示任意一个字符。

使用‘^’在字符序列第一个表示字符开始符号。
使用‘$’在字符序列最后一个表示字符结束符号。
使用‘\\b’匹配单词边界，即左右两边至少有一个不是单词，单词即为‘\\w’，即字母、数字和下划线。
使用‘\\B’匹配非单词边界，即左右两边都是单词。
非单词不是边界、非单词与单词之间的才叫边界。

#### 1.1.2 析取、分组和优先级

disjunction：使用管道符‘|’，即cat|dog可以匹配cat或者dog。
precedence：使用‘（）’配合‘|’，即gupp(y|ies)可以匹配guppy和guppies。
counter

### 1.2 Words

### 1.3 Corpora（任何事物之主体、全集）

### 1.4 Text Normalization

### 1.5 Minimum Edit Distance

### 1.6 Summary

## 2. N-gram Language Models

## 3. Naive Bayes and Sentiment Classification（朴素贝叶斯与情感分类）

## 4. Logistic Regression

## 5. Vector Semantics and Embeddings（向量语义和嵌入）

## 6. Neural Networks and Neural Language Models 

## 7. Sequence Labeling for Parts of Speech and Named Entities（词性和命名实体识别的序列标注）

## 8. Deep Learning Architectures for Sequence Processing（序列处理的深度学习架构）

## 9. Machine Translation and Encoder-Decoder Models

## 10. Transfer Learning with Pretrained Language Models and Contextual Embeddings（基于预训练语言模型和上下文嵌入的迁移学习）

## 11. Constituency Grammars（选区语法）

## 12. Constituency Parsing（选取解析）

## 13. Dependency Parsing

## 14. Logical Representations of Sentence Meaning（句子意义的逻辑表征）

## 15. Computational Semantics and Semantic Parsing（计算语义和语义解析）

## 16. Information Extraction

## 17. Word Senses and WordNet

## 18. Semantic Role Labeling

## 19. Lexicons for Sentiment, Affect, and Connotation

## 20. Coreference Resolution（算法）

## 21. Discourse Coherence（语篇连贯）

## 22. Question Answering

## 23. Chatbots & Dialogue Systems（聊天机器人和对话系统）

## 24. Phonetics（语音学）

## 25. Automatic Speech Recognition and Text-to-Speech
