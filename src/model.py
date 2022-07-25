"""
Chapter 2. Case Study
"""
from collections.abc import Iterator
import abc
import collections
import datetime
from typing import (
    cast,
    Any,
    Optional,
    Union,
    Iterable,
    Counter,
    Callable,
    Protocol,
    List,
    Dict,
    Tuple
)
from math import isclose, hypot


class Sample:
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f")"
        )


class KnownSample(Sample):
    """Abstract superclass for testing and training data, the species is set externally."""

    def __init__(
            self,
            species: str,
            sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float,
    ) -> None:
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.species = species

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}, "
            f")"
        )


class TrainingKnownSample(KnownSample):
    """Training data."""
    pass


class TestingKnownSample(KnownSample):
    """Testing data. A classifier can assign a species, which may or may not be correct."""

    def __init__(
            self,
            species: str,
            sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float,
            classification: Optional[str] = None,
    ) -> None:
        super().__init__(
            species=species,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.classification = classification

    def matches(self) -> bool:
        return self.species == self.classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}, "
            f"classification={self.classification!r}, "
            f")"
        )


class UnknownSample(Sample):
    """A sample provided by a User, not yet classified."""
    pass


class ClassifiedSample(Sample):
    """Created from a sample provided by a User, and the results of classification."""

    def __init__(self, classification: str, sample: UnknownSample) -> None:
        super().__init__(
            sepal_length=sample.sepal_length,
            sepal_width=sample.sepal_width,
            petal_length=sample.petal_length,
            petal_width=sample.petal_width,
        )
        self.classification = classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"classification={self.classification!r}, "
            f")"
        )


class Distance:
    """Definition of a distance computation"""

    def distance(self, s1: Sample, s2: Sample) -> float:
        pass


class ED(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width
        )

class MD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width)
            ]
        )

class CD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width)
            ]
        )

class SD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width)
            ]
        ) / sum(
            [
                abs(s1.sepal_length + s2.sepal_length),
                abs(s1.sepal_width + s2.sepal_width),
                abs(s1.petal_length + s2.petal_length),
                abs(s1.petal_width + s2.petal_width)
            ]
        )

class Chebyshev(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_width),
                abs(s1.petal_width - s2.petal_width),
            ]
        )

class Minkowski(Distance):
    @property
    @abc.abstractmethod
    def m(self) -> int:
        ...

    def distance(self, s1: Sample, s2: Sample) -> float:
        return (
            sum(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m
                ]
            ) ** (1 / self.m)
        )

class Euclidean(Minkowski):
    m = 2

class Manhattan(Minkowski):
    m = 1

class Sorensen(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        ) / sum (
            [
                s1.sepal_length + s2.sepal_length,
                s1.sepal_width + s2.sepal_width,
                s1.petal_length + s2.petal_length,
                s1.petal_width + s2.petal_width,
            ]
        )

class Minkowski_2(Distance):
    @property
    @abc.abstractmethod
    def m(self) -> int:
        ...

    @staticmethod
    @abc.abstractstaticmethod
    def reduction(values: Iterable[float]) -> float:
        ...

    def distance(self, s1: Sample, s2: Sample) -> float:
        return (
            self.reduction(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m,
                ]
            ) ** (1 / self.m)
        )

class CD2(Minkowski_2):
    m = 1

    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return max(values)

class MD2(Minkowski_2):
    m = 1
    
    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return sum(values)

class ED2(Minkowski_2):
    m = 2

    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return sum(values)

class ED2S(Minkowski_2):
    m = 2
    reduction = sum # type: ignore [assignment]

class Hyperparamter:
    """
    A hyperparamter value and the overall quality of the classification.
    """

    def __init__(self, k: int, algorithm: "Distance", training: "TrainingData") -> None:
        self.k = k
        self.algorithm = algorithm
        self.data: TrainingData = training
        self.quality: float

    def test(self) -> None:
        """Runs the entire test suite."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken Weak Reference")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Union[UnknownSample, TestingKnownSample]) -> str:
        """TODO: the k-NN algorithm"""
        training_data = self.data()
        if not training_data:
            raise RuntimeError("No TrainingData object")
        distances: List[Tuple[float, TrainingKnownSample]] = sorted(
            (self.algorithm.distance(sample, known), known)
            for known in training_data.training
        )
        k_nearest = (known.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *other = frequency.most_common()
        species, votes = best_fit
        return species


class TrainingData:
    """A set of training data and testing data with methods to load and test the samples."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: List[Sample] = []
        self.testing: List[Sample] = []
        self.tunning: List[Hyperparamter] = []

    def load(self, raw_data_source: Iterable[Dict[str, str]]) -> None:
        """Load and partition the raw data"""
        for n, row in enumerate(raw_data_source):
            sample = Sample(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
                species=row["species"],
            )
            if n % 5 == 0:
                self.testing.append(sample)
            else:
                self.training.append(sample)

        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparamter) -> None:
        """Test this hyperparameter value."""
        parameter.test()
        self.tunning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(
        self, parameter: Hyperparamter, sample: UnknownSample
    ) -> ClassifiedSample:
        return ClassifiedSample(
            classification=parameter.classify(sample), sample=sample
        )