import importlib.metadata

project = "ecpfs"
author = "Omar Shahbaz Khan"
release = importlib.metadata.version("ecpfs")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"

html_theme = "shibuya"
