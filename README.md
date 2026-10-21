# mynote

将课程课件或学习材料整理为贴合原文结构的 Markdown 学习笔记的可复用 Agent Skill。

## 特点

- 英文要点配必要中文释义，难点按步骤解释。
- 依据原材料组织内容，保留用户最新修改。
- 精准裁切课件配图，保留坐标、图例与必要标签。
- 附完整课件—笔记对照及配图，可按需参考。

## 安装与使用

将本仓库克隆到 Codex 的个人技能目录（目标目录不存在时）：

```bash
git clone https://github.com/CrRdz/mynote.git ~/.codex/skills/mynote
```

如果已有同名技能，请先保留自己的修改，再更新目录。

示例请求：

> 使用 $mynote，帮我将这份课件整理成笔记，在必要处补中文解释，配图只保留相关区域。

## 文件

- [SKILL.md](SKILL.md)：跨课程的通用规则。
- [风格示例](references/style-examples.md)：精简写法参考。
- [课件与笔记对照导读](references/paired-example/guide.md)：源课件、笔记快照和配图的对应关系。
- [界面配置](agents/openai.yaml)：技能展示信息。

参考材料用于示范整理方法，不限定新任务的学科或结构。课件原作者信息保留在 PDF 中；本仓库未对第三方课程材料授予额外许可。
