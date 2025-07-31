try:
    from minions.clients.ollama import OllamaClient
except:
    print("OllamaClient is not installed. Please install it using `pip install ollama`.")

try:
    import chromadb
except:
    print("chromadb is not installed. Please install it using `pip install chromadb`.")

from datetime import datetime
import os
import traceback
from typing import Optional, Union, List, Dict, Any, Tuple
import functools


def clear_system_cache_after(func):
    """
    Function decorator to clear ChromaDB system cache.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # chromadb.api.client.SharedSystemClient.clear_system_cache()
        return result

    return wrapper


class TextEmbedding:
    def __init__(self, embedding, text_body, file_path="") -> None:
        self.embedding = embedding
        self.content = text_body
        self.content_path = file_path

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TextEmbedding to dictionary representation.

        :returns Dict representation of the TextEmbedding
        """
        return {
            "content": self.content,
            "type": "text",
            "content_path": self.content_path,
        }


class ImageEmbedding:
    def __init__(self, embedding, image_path, upload, frame_id=0) -> None:
        self.embedding = embedding
        self.content_path = image_path
        self.upload = upload
        self.frame_id = frame_id

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ImageEmbedding to dictionary representation.

        :returns Dict representation of the ImageEmbedding
        """
        return {
            "content_path": self.content_path,
            "type": "image",
            "upload": self.upload,
            "frame_id": self.frame_id,
        }


class VideoEmbedding:
    def __init__(
        self, frame_embeddings: List[ImageEmbedding], video_path, upload
    ) -> None:
        self.frame_embeddings = frame_embeddings
        self.content_path = video_path
        self.upload = upload

        raise Exception("VideoEmbedding not supported yet!")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert VideoEmbedding to dictionary representation.

        :returns Dict representation of the VideoEmbedding
        """

        return {
            "content_path": self.content_path,
            "type": "video",
            "upload": self.upload,
            "frame_ids": [frame.frame_id for frame in self.frame_embeddings],
        }


class MultiModalEmbedder:
    def __init__(self, model_name: str = "llava") -> None:
        """
        Initialize the MultiModalEmbedder.

        :param model_name: name of the local ollama model used to embed text, images, etc.
        """
        self.model_name = model_name
        self.client = OllamaClient(
            model_name=model_name,
            use_async=False,
            num_ctx=131072,  # what is this?
        )

    def _embed_image(self, content, image_path, upload):
        """
        Generate embeddings for an image using the ollama model.

        Theoretically, OllamaClient.embed could directly be called from self.generate_embedding
        but this is cleaner.

        :param content: image body to be embedded
        :param image_path: local path/s3 uri image
        :param upload: whether image_path points to s3 bucket or local file
        :returns ImageEmbedding Object
        """

        # do type sanity checks
        ...

        # generate embedding
        embedding = self.client.embed(content=content)
        return ImageEmbedding(
            embedding=embedding[0], image_path=image_path, upload=upload
        )

    def _embed_text(self, text, path=""):
        """
        Generate embeddings for an image using the ollama model.

        Theoretically, OllamaClient.embed could directly be called from self.generate_embedding
        but this is cleaner.

        :param text: text body to be embedded
        :returns TextEmbedding Object
        """
        # generate embeddings
        embedding = self.client.embed(content=text)
        return TextEmbedding(embedding=embedding[0], text_body=text, file_path=path)

    def _embed_video(self, content, video_path, upload):
        """
        Generate embeddings for a video using the ollama model.
        Currently, an embedding of a video is represented by a set of embeddings for frames
        randomly selected from a video. Other approaches may include:
            - aggregating (mean, sum) of randomly selected frame embeddings
            - native multi-modal video/text/image model
            - ... (we have a lot of room for creativity here!)

        :param content: video body to be embedded
        :param video_path: local path/s3 uri video
        :param upload: whether video_path points to s3 bucket or local file
        :returns VideoEmbedding Object
        """

        # do type sanity checks

        # select random frames
        frames = []
        frame_ids = []

        # generate embeddings
        frame_embeddings = []
        for frame, frame_id in zip(frames, frame_ids):
            embedding = self.client.embed(content=frame)
            frame_embeddings.append(
                ImageEmbedding(
                    embedding=embedding[0],
                    image_path=video_path,
                    upload=True,
                    frame_id=frame_id,
                )
            )
        return VideoEmbedding(
            frame_embeddings=frame_embeddings, video_path=video_path, upload=upload
        )

    def _upload_to_s3(self, content, content_type: str, s3_bucket: str) -> str:
        """
        Upload content of content_type to specified s3 bucket.

        :param content: media body to be uploaded
        :param content_type: type of media to be uploaded
        :param s3_bucket: name of the s3 bucket for file upload
        :returns uri to file in s3 bucket, None if unsuccessful
        """
        raise Exception("_upload_to_s3 not supported yet!")

    def generate_embedding(
        self,
        content,
        content_type: str = "text",
        upload: bool = False,
        path: str = None,
    ):
        if content_type == "text":
            return self._embed_text(content, path=path)
        elif content_type in ["image", "video"]:
            # we need to store the images somewhere, either local path or upload
            # if we want to use the local path, a path must be provided
            # if upload==True, upload to a s3 bucket we are hosting
            if not upload and not path:
                raise Exception("Must provide a local path to media if upload=False")
            if upload:
                path = self._upload_to_s3(
                    content, content_type, s3_bucket=os.environ.get("S3_BUCKET_MEDIA")
                )
                if not path:
                    raise Exception("Upload unsuccessful!")

            # embed content
            if content_type == "image":
                return self._embed_image(content, path, upload)
            else:
                return self._embed_video(content, path, upload)
        else:
            raise Exception(
                f"Content Type {content_type} not supported yet! MultiModalEmbedder supports ['text', 'image', 'video']"
            )


class ChromaDBCollection:
    def __init__(
        self,
        embedding_model: str = "llava",
        collection_name: Optional[str] = None,
        dev=False,
    ) -> None:
        """
        Initialize a ChromaDB client and collection.

        :param embedding_model: name of the local ollama model used to embed text, images, etc.
        :param collection_name: name of the collection/index this client should create, defaults
            to a hash of the current date timestamp and embedding model
        :param dev: whether the ChromaDB collection is used for development purposes, added to metadata
        """
        # chromadb.api.client.SharedSystemClient.clear_system_cache()
        # default collection name to hash
        if collection_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            collection_name = f"{embedding_model}_{timestamp}"
            print(
                f"No collection name provided, using generated name: {collection_name}"
            )

        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.dev = dev

        self.client = chromadb.PersistentClient()
        try:
            print(f"Trying to fetch collection {collection_name}.")
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            print(f"Collection {collection_name} doesn't exist, creating a new one.")
            self.collection = self.client.create_collection(
                collection_name,
                metadata={"embedding_model": embedding_model, "dev": self.dev},
            )

        self.exists = True  # track whether the collection still exists

    def _embedding_from_result(
        self, embedding, metadata
    ) -> Union[TextEmbedding, ImageEmbedding, VideoEmbedding]:
        """Converts ChromaDB query result to Embedding Object."""
        metadata = metadata[0]
        embedding_type = metadata["type"]

        if embedding_type == "text":
            return TextEmbedding(
                embedding=embedding,
                text_body=metadata["content"],
                file_path=metadata["content_path"],
            )
        elif embedding_type == "image":
            return ImageEmbedding(
                embedding=embedding,
                image_path=metadata["content_path"],
                upload=metadata["upload"],
            )
        elif embedding_type == "video":
            return VideoEmbedding(
                embedding=embedding,
                video_path=metadata["content_path"],
                upload=metadata["upload"],
            )
        else:
            return None

    @clear_system_cache_after
    def delete_collection(self) -> bool:
        """
        Delete the current collection from ChromaDB.

        :returns Whether the collection was successfully deleted
        """
        try:
            if self.exists:
                self.client.delete_collection(self.collection_name)
                self.collection = None
                self.exists = False
                return True
            else:
                print(f"No collection {self.collection_name} to delete.")
                return False
        except Exception as e:
            print(f"Error deleting collection {self.collection_name}:")
            traceback.print_exc()
            return False

    @clear_system_cache_after
    def add_entries(
        self,
        embeddings: List[Union[TextEmbedding, ImageEmbedding, VideoEmbedding]],
    ) -> List[str]:
        """
        Add entries to the ChromaDB collection.

        :param embeddings: list of document embeddings to add, must be either TextEmbedding, ImageEmbedding, or
            VideoEmbedding object
        :returns List of ids of added entries
        """
        embeddings_vecs = [embedding.embedding for embedding in embeddings]

        # metadata is stored in embeddings:
        metadata = []
        for embedding in embeddings:
            embedding_metadata = embedding.to_dict()
            embedding_metadata["model"] = self.embedding_model
            metadata.append(embedding_metadata)

        # need to specify ids, doesn't have to have semantic meaning since metadata
        # should include all the important information we need
        import uuid

        ids = [str(uuid.uuid4()) for _ in embeddings]

        self.collection.add(embeddings=embeddings_vecs, metadatas=metadata, ids=ids)

        return ids

    @clear_system_cache_after
    def retrieve(
        self, query: TextEmbedding, top_k: int
    ) -> List[Tuple[int, Union[TextEmbedding, ImageEmbedding, VideoEmbedding]]]:
        """
        Retrieve entries to the ChromaDB collection based on query.

        :param query: embedding object (TextEmbedding) of query
        :param top_k: number of top K documents to query from ChromaDB collection
        :returns List of top N most relevant documents and their distance to the query in a tuple, either
            TextEmbedding, ImageEmbedding, or VideoEmbedding object
        """
        # NOTE: don't have to necessarily query with TextEmbedding, but use case currently
        # limited to text querying

        results = self.collection.query(
            query_embeddings=[query.embedding],
            n_results=top_k,
            include=["embeddings", "metadatas", "distances"],
        )

        return [
            (self._embedding_from_result(embedding, metadata), distance)
            for embedding, metadata, distance in zip(
                results["embeddings"], results["metadatas"], results["distances"]
            )
        ]

    @clear_system_cache_after
    def collection_size(self) -> int:
        """Return number of entries in current collection."""
        return self.collection.count()


@clear_system_cache_after
def embed_and_add(
    chromadb: ChromaDBCollection, content: str, content_type: str, upload=False, path=""
) -> List[str]:
    """
    Embed content and add embeddings to the ChromaDB collection.

    :param chromadb: ChromaDBCollection instance to add embeddings to
    :param content: content to embed (text, image bytes, or video bytes)
    :param content_type: type of content ('text', 'image', or 'video')
    :param upload: whether to upload media content to S3 (only applicable for image/video)
    :param path: local path for media content (required if upload=False for image/video)
    :returns Embedding Object and ids of added entry
    """
    embedder = MultiModalEmbedder(model_name=chromadb.embedding_model)

    embedding = embedder.generate_embedding(
        content=content, content_type=content_type, upload=upload, path=path
    )

    # Add to ChromaDB collection
    return embedding, chromadb.add_entries([embedding])


@clear_system_cache_after
def embed_and_retrieve(chromadb: ChromaDBCollection, query_text: str, top_k: int = 1):
    """
    Embed query string find most relevant entries from the ChromaDB collection.

    :param chromadb: ChromaDBCollection instance to add embeddings to
    :param query_text: query to search collection by
    :param content_type: type of query_body ('text', 'image', or 'video')
    :param top_k: number of most relevant entries to retrieve
    :returns List of top_k most relevant Embedding Objects
    """
    embedder = MultiModalEmbedder(model_name=chromadb.embedding_model)
    query_embedding = embedder.generate_embedding(
        content=query_text, content_type="text"
    )
    # Retrieve from ChromaDB collection
    results = chromadb.retrieve(query=query_embedding, top_k=top_k)
    return results


def retrieve_chunks_from_chroma(
    chunks, keywords, embedding_model="granite3.2-vision", k=10
):
    collection = ChromaDBCollection(embedding_model=embedding_model)

    # TODO: batch operation
    for i, chunk in enumerate(chunks):
        # Embed and add each chunk to the collection
        embed_and_add(collection, content=chunk, content_type="text", path="")
    # construct query by concatenating all keywords
    query = " ".join(keywords)

    search_results = embed_and_retrieve(collection, query_text=query, top_k=k)
    relevant_chunks = []
    for result in search_results:
        relevant_chunks.append(result.content)
    return relevant_chunks
