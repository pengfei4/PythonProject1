# Project: [项目名]

## 技术栈
- 前端: React 18 + TypeScript 5
- 后端: Node.js 20 + Fastify
- 数据库: PostgreSQL 16
- ORM: Drizzle

## 代码规范
- 使用单引号
- 缩进2空格
- 函数式组件优先
- 禁止any类型

## 项目结构
src/
├── components/  # React组件
├── hooks/       # 自定义Hooks
├── services/    # API调用
└── utils/       # 工具函数

## 测试要求
- 新功能必须有单元测试
- 运行 `npm test` 验证
- 覆盖率不低于80%

## 禁止事项
- 不要修改 .env 文件
- 不要直接操作数据库
- 不要删除 node_modules

## 沟通风格
- 简洁，不废话
- 遇到不确定的先问
- 给出方案时说明取舍
