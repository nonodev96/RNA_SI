# Manual

<https://www.pragmaticlinux.com/2021/01/create-a-man-page-for-your-own-program-or-script-with-pandoc/>

```bash
sudo apt install pandoc
```

```bash
man man
```

## Create the MAN page for the script with Pandoc

[PROGRAM NAME].[SECTION NUMBER]

## MAN page framework in Markdown

```md
---
title: MANUAL
section: 1
header: User Manual
footer: manual 1.0.0
date: January 20, 2024
---
```

```bash
pandoc hello.1.md -s -t man -o hello.1
```

## Using MANPATH

```bash
manpath
```
