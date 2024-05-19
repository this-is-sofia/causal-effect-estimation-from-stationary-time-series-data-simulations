def fix_node_names_for_title(name):
    if isinstance(name, str):
        name_parts = name.split("-")
        return "_".join(name_parts[:-1]) + "_" + name_parts[-1].replace("t", "t+")
    elif isinstance(name, list):
        return f'[{", ".join([fix_node_names_for_title(name_part) for name_part in name])}]'
