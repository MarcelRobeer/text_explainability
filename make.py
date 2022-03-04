
#!/usr/bin/env python3
import shutil
from pathlib import Path

from pdoc import pdoc, render

here = Path(__file__).parent
out = here / "docs_test"

# Render pdoc's documentation into docs/api...
render.configure(template_directory=here / "templates")
pdoc("text_explainability", output_directory=out)

# ...and rename the .html files to .md so that mkdocs picks them up!
for f in out.glob("**/*.html"):
    f.rename(f.with_suffix(".md"))

