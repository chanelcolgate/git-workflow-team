"""
Chapter 2. Case Study
"""
from collections.abc import Iterator
import datetime
from typing import Optional, Union, Iterable, List, Dict

class Sample:
	def __init__(
		self,
		sepal_length: float,
		sepal_width: float,
		petal_length: float,
		petal_width: float,
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

class Hyperparamter:
	"""
	A hyperparamter value and the overall quality of the classification.
	"""
	def __init__(self, k: int, training: "TrainingData") -> None:
		self.k = k
		self.data: TrainingData = training
		self.quality: float

	def test(self) -> None:
		"""Runs the entire test suite."""
		pass_count, fail_count = 0, 0
		for sample in self.data.testing:
			sample.classification = self.classify(sample)
			if sample.matches():
				pass_count += 1
			else:
				fail_count += 1
		self.quality = pass_count / (pass_count + fail_count)

	def classify(self, sample: Sample) -> str:
		"""TODO: the k-NN algorithm"""
		return ""

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