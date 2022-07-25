from http.client import HTTPException
from typing import List, Tuple

from fastapi import Depends, FastAPI, Query, status
from tortoise.contrib.fastapi import register_tortoise
from tortoise.exceptions import DoesNotExist

from src.models import (
	PostDB,
	PostCreate,
	PostPartialUpdate,
	PostTortoise,
	PostPublic,
	CommentDB,
	CommentBase,
	CommentTortoise
)

app = FastAPI()

async def pagination(
	skip: int = Query(0, ge=0),
	limit: int = Query(10, ge=0),
) -> Tuple[int, int]:
	capped_limit = min(100, limit)
	return (skip, capped_limit)

async def get_post_or_404(id: int) -> PostTortoise:
	try:
		return await PostTortoise.get(id=id).prefetch_related("comments")
	except DoesNotExist:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

@app.get("/posts")
async def list_posts(pagination: Tuple[int, int] = Depends(pagination)) -> List[PostDB]:
	skip, limit = pagination
	posts = await PostTortoise.all().offset(skip).limit(limit)
	results = [PostDB.from_orm(post) for post in posts]
	return results

@app.get("/posts/{id}", response_model=PostDB)
async def get_post(post: PostTortoise = Depends(get_post_or_404)) -> PostPublic:
	return PostPublic.from_orm(post)

@app.post("/posts", response_model=PostDB, status_code=status.HTTP_201_CREATED)
async def create_post(post: PostCreate) -> PostPublic:
	post_tortoise = await PostTortoise.create(**post.dict())
	return post_tortoise.fetch_related("comments")

@app.patch("/posts/{id}", response_model=PostDB)
async def update_post(
	post_update: PostPartialUpdate, post: PostTortoise = Depends(get_post_or_404)
) -> PostDB:
	post.update_from_dict(post_update.dict(exclude_unset=True))
	await post.save()

@app.delete("/posts/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(post: PostTortoise = Depends(get_post_or_404)):
	await post.delete()

TORTOISE_ORM = {
	"connections": {"default": "postgres://postgres:postgres@iotmind.ddns.net:49200/memory"},
	"apps": {
		"models": {
			"models": ["src.models"],
			"default_connection": "default",
		},
	},
}

register_tortoise(
	app,
	config=TORTOISE_ORM,
	generate_schemas=True,
	add_exception_handlers=True,
)