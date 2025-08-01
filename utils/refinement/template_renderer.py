from string import Template


class TemplateRenderer:
    """Render question templates using string.Template."""

    def render(self, template_str: str, **kwargs: str) -> str:
        return Template(template_str).safe_substitute(**kwargs)
