import attr
from itertools import product


@attr.s(auto_attribs=True)
class Expt:
    kernel_stride: float
    instances: int
    gpus: int

    def __str__(self):
        attrs = []
        for a in self.__attrs_attrs__:
            attrs.append("{}={}".format(a.name, self.__dict__[a.name]))
        return "-".join(attrs)


kernel_strides = [0.001, 0.002, 0.004, 0.1]
instances = [1, 2, 4]
gpus = [1, 2, 4]


expts = []
for k, i, g in product(kernel_strides, instances, gpus):
    expts.append(Expt(k, i , g))
