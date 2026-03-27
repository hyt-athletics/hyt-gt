from enum import StrEnum


class Method(StrEnum):
    ERT = "ert"
    IP = "ip"
    EM = "em"
    TEM = "tem"
    LOGGING = "logging"

    @property
    def label(self) -> str:
        return _METHOD_LABELS[self]


class Category(StrEnum):
    PREPROCESS = "preprocess"
    PROCESS = "process"
    FORWARD = "forward"
    INVERSION = "inversion"

    @property
    def label(self) -> str:
        return _CATEGORY_LABELS[self]


_METHOD_LABELS = {
    Method.ERT: "ERT 电阻率法",
    Method.IP: "IP 激发极化法",
    Method.EM: "EM 井间电磁波法",
    Method.TEM: "TEM 瞬变电磁法",
    Method.LOGGING: "金属矿测井",
}

_CATEGORY_LABELS = {
    Category.PREPROCESS: "预处理",
    Category.PROCESS: "处理",
    Category.FORWARD: "正演",
    Category.INVERSION: "反演",
}

METHOD_ORDER: list[Method] = [Method.ERT, Method.IP, Method.EM, Method.TEM, Method.LOGGING]
CATEGORY_ORDER: list[Category] = [
    Category.PREPROCESS, Category.PROCESS, Category.FORWARD, Category.INVERSION,
]
