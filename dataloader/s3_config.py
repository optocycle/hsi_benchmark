from dataclasses import dataclass
from minio import Minio
import json

@dataclass
class S3Config:
    endpoint: str
    credentials_file: str = "/home/jovan/.minio/credentials.json"

    @property
    def access_key(self) -> str:
        self._load_credentials()
        return self.__loaded_creds.get("accessKey")

    @property
    def secret_key(self) -> str:
        self._load_credentials()
        return self.__loaded_creds.get("secretKey")

    def _load_credentials(self) -> None:
        if hasattr(self, "__loaded_creds"):
            return
        with open(self.credentials_file) as fp:
            self.__loaded_creds = json.load(fp)

    def get_instance(self, region: str=None) -> Minio:
        s3 = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=region
        )
        return s3
