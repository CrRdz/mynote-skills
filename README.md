# mynote

将课件整理成 Markdown 学习笔记的 Codex Skill：英文要点配中文解释，难点分步推导，配图只裁取相关区域。附完整课件与笔记对照，适用于不同课程。

## 安装

在终端执行：

```bash
mkdir -p ~/.codex/skills
git clone https://github.com/CrRdz/mynote-skills.git ~/.codex/skills/mynote
```

若同名目录已存在，请先备份或移走，避免覆盖自己的修改。

## 使用

提供课件文件或路径，在 Codex 中输入：

> 使用 $mynote，帮我整理这份课件，必要处加中文解释，配图裁切对应区域。

也可提供已有笔记作为风格参考。

## 更新

通过上述方式安装后，可执行：

```bash
git -C ~/.codex/skills/mynote pull --ff-only
```

参考课件的权利归原作者所有。
