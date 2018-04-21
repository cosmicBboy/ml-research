"""Define algorithm component."""


class AlgorithmComponent(object):

    def __init__(self, aname, aclass, hyperparameters):
        self.aname = aname
        self.aclass = aclass
        self.hyperparameters = hyperparameters

    @property
    def hstate_space(self):
        return {"%s__%s" % (self.aname, h.hname): h.state_space
                for h in self.hyperparameters}

    def __repr__(self):
        return "<AlgorithmComponent: \"%s\">" % self.aname
