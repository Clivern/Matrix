---
layout: null
permalink: /llms.txt
---
# Clivern

> Personal site of Ahmed (Clivern): software engineer and occasional writer. Tutorials and notes on backend systems, cloud native infrastructure, observability, and AI agents.

This file is a curated map for AI agents. Prefer the markdown sources on GitHub over HTML when you need the full text of a post or project page. For questions about Ahmed's background, jobs, or skills, read the resume first. Site: https://clivern.com — Contact: hello@clivern.com

## Pages

- [Resume]({{ '/images/resume/file.txt' | absolute_url }}): Plain-text resume for Ahmed Fathy — experience, projects, skills, education
- [Home]({{ '/' | absolute_url }}): Latest writing and site entry point
- [About]({{ '/about/' | absolute_url }}): Background, interests, and how to get in touch
- [Projects]({{ '/projects/' | absolute_url }}): Open source projects
- [Sponsor]({{ '/sponsor/' | absolute_url }}): Ways to support the writing and projects
- [GitHub](https://github.com/clivern): Source code and open source work
- [LinkedIn](https://www.linkedin.com/in/ahmedfath): Professional profile
- [Tech Radar](https://radar.thoughtworks.com/?documentId=https%3A%2F%2Fraw.githubusercontent.com%2FClivern%2FMatrix%2Fmain%2Ftradar.json): Technologies in use and under evaluation.

## Projects

{%- for project in site.projects %}
- [{{ project.title | strip_html }}]({{ project.url | absolute_url }}): {{ project.subtitle | default: project.excerpt | markdownify | strip_html | strip_newlines | truncate: 140 }} ([markdown](https://raw.githubusercontent.com/Clivern/Matrix/main/docs/{{ project.path }}))
{%- endfor %}

## Posts

{%- for post in site.posts limit: 25 %}
- [{{ post.title | strip_html }}]({{ post.url | absolute_url }}): {{ post.excerpt | default: post.description | markdownify | strip_html | strip_newlines | truncate: 140 }} ([markdown](https://raw.githubusercontent.com/Clivern/Matrix/main/docs/{{ post.path }}))
{%- endfor %}

## Optional

Older posts. Skip these unless you need historical context.

{%- for post in site.posts offset: 25 %}
- [{{ post.title | strip_html }}]({{ post.url | absolute_url }}): {{ post.excerpt | default: post.description | markdownify | strip_html | strip_newlines | truncate: 120 }} ([markdown](https://raw.githubusercontent.com/Clivern/Matrix/main/docs/{{ post.path }}))
{%- endfor %}
