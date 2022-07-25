"""
This example shows how relations between models work.

Key points in this example are use of ForeignKeyField and ManyToManyField
to declare relations and use of .prefetch_related() and .fetch_related()
to get this related objects
"""
from tortoise import Tortoise, fields, run_async
from tortoise.exceptions import NoValuesFetched
from tortoise.models import Model

class Tournament(Model):
	id = fields.IntField(pk=True)
	name = fields.TextField()

	events: fields.ReverseRelation["Event"]

	def __str__(self):
		return self.name

class Event(Model):
	id = fields.IntField(pk=True)
	name = fields.TextField()
	tournament: fields.ForeignKeyRelation[Tournament] = fields.ForeignKeyField(
		"models.Tournament", related_name="events"
	)
	participants: fields.ManyToManyRelation["Team"] = fields.ManyToManyField(
		"models.Team", related_name="events", through="event_team"
	)

	def __str__(self):
		return self.name

class Address(Model):
	city = fields.CharField(max_length=64)
	street = fields.CharField(max_length=128)

	event: fields.OneToOneRelation[Event] = fields.OneToOneField(
		"models.Event", on_delete=fields.CASCADE, related_name="address", pk=True
	)

	def __str__(self):
		return f"Address({self.city}, {self.street})"

class Team(Model):
	id = fields.IntField(pk=True)
	name = fields.TextField()

	events: fields.ManyToManyRelation[Event]

	def __str__(self):
		return self.name