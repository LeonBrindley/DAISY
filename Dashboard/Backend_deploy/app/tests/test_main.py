from uuid import UUID
from starlette import status
from starlette.testclient import TestClient
from app.repositories import ImageRepository
from app.schemas import ImageSchemaIn

def test_data_create(client: TestClient) -> None:
    data = {
        "img_url": "http://example.com/image.jpg",
        "coordinates": [123, 456],
        "time": 1625260800,
    }

    response = client.post("/v1/data", json=data)

    assert response.status_code == status.HTTP_201_CREATED

    response_data = response.json()
    assert response_data["img_url"] == data["img_url"]
    assert response_data["coordinates"] == data["coordinates"]
    assert response_data["time"] == data["time"]
    assert isinstance(response_data["id"], str)
    assert isinstance(response_data["created_at"], int)
    assert isinstance(response_data["updated_at"], int)

    # Check if data exists in db
    stored_data = DataRepository.get(response_data["id"])
    print("The ID is: ", response_data["id"])
    assert stored_data is not None
    assert stored_data.img_url == data["img_url"]
    assert stored_data.coordinates == data["coordinates"]
    assert stored_data.time == data["time"]

def test_data_get(client: TestClient) -> None:
    data_in = DataSchemaIn(
        img_url="http://example.com/image.jpg",
        coordinates=[123, 456],
        time=1625260800,

    )
    data = DataRepository.create(data_in)

    response = client.get(f"/v1/data/{data.id}")

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert response_data == {
        "id": str(data.id),
        "img_url": data.img_url,
        "coordinates": data.coordinates,
        "time": data.time,
        "created_at": data.created_at,
        "updated_at": data.updated_at,
    }
