from typing import TYPE_CHECKING, Dict, List, Set


if TYPE_CHECKING:
    from gradio.components import Component


class Manager:
    def __init__(self) -> None:
        self.all_elems: Dict[str, Dict[str, "Component"]] = {}

    def get_elem_by_name(self, name: str) -> "Component":
        r"""
        Example: top.lang, train.dataset
        """
        tab_name, elem_name = name.split(".")
        return self.all_elems[tab_name][elem_name]

    def list_elems(self) -> List["Component"]:
        return [elem for elems in self.all_elems.values() for elem in elems.values()]
