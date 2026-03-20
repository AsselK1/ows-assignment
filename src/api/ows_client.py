from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Mapping
from typing import cast

import requests

LOGGER = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
CLIENT_ERROR_STATUS_CODES = {400, 401, 403, 404, 429}

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | dict[str, "JSONValue"] | list["JSONValue"]
JSONObject = dict[str, JSONValue]


class OWSAPIError(Exception):
    pass


class OWSAuthError(OWSAPIError):
    pass


class OWSRateLimitError(OWSAPIError):
    pass


class BaseOWSClient:
    base_url: str
    _session: requests.Session
    _last_request_time: float

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str = "https://ows.goszakup.gov.kz/v3",
    ) -> None:
        token = api_token or os.getenv("OWS_API_TOKEN")
        if not token:
            raise OWSAuthError("OWS API token is required. Set OWS_API_TOKEN environment variable.")

        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self._last_request_time = 0.0

        masked_token = f"{token[:8]}..." if len(token) >= 8 else "***"
        LOGGER.info("OWS client initialized for %s with token %s", self.base_url, masked_token)

    def _build_url(self, endpoint: str) -> str:
        normalized = endpoint.strip()
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        if normalized.startswith("/v3/"):
            normalized = normalized[3:]
        return f"{self.base_url}{normalized}"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    def _raise_for_client_error(self, status_code: int, response_text: str) -> None:
        if status_code == 401:
            raise OWSAuthError("Authentication failed: invalid or expired token.")
        if status_code == 429:
            raise OWSRateLimitError("OWS API rate limit exceeded.")
        if status_code in CLIENT_ERROR_STATUS_CODES or 400 <= status_code < 500:
            raise OWSAPIError(f"OWS API client error {status_code}: {response_text}")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Mapping[str, JSONScalar] | None = None,
        json: Mapping[str, JSONValue] | None = None,
    ) -> JSONObject:
        url = self._build_url(endpoint)
        max_retries = 3
        backoffs = [1.0, 2.0, 4.0]
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            self._throttle()
            try:
                response = self._session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json,
                    timeout=30,
                )
                self._last_request_time = time.monotonic()

                status_code = response.status_code
                if status_code in RETRYABLE_STATUS_CODES:
                    if attempt < max_retries:
                        wait_seconds = backoffs[attempt]
                        LOGGER.warning(
                            "OWS server error %s on %s %s, retrying in %.1fs",
                            status_code,
                            method.upper(),
                            endpoint,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                        continue
                    raise OWSAPIError(
                        f"OWS API server error {status_code} after {max_retries + 1} attempts"
                    )

                self._raise_for_client_error(status_code, response.text)

                try:
                    payload = cast(object, response.json())
                except ValueError as error:
                    raise OWSAPIError(
                        f"OWS API returned non-JSON response for {method.upper()} {endpoint}"
                    ) from error

                if not isinstance(payload, dict):
                    raise OWSAPIError(
                        f"OWS API response for {method.upper()} {endpoint} is not a JSON object"
                    )

                payload_object = cast(JSONObject, payload)
                return payload_object
            except (OWSAuthError, OWSRateLimitError, OWSAPIError):
                raise
            except requests.RequestException as error:
                last_error = error
                if attempt < max_retries:
                    wait_seconds = backoffs[attempt]
                    LOGGER.warning(
                        "Network error on %s %s, retrying in %.1fs: %s",
                        method.upper(),
                        endpoint,
                        wait_seconds,
                        error,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise OWSAPIError(
                    f"OWS API request failed after {max_retries + 1} attempts: {error}"
                ) from error

        if last_error is not None:
            raise OWSAPIError(f"OWS API request failed: {last_error}") from last_error
        raise OWSAPIError("OWS API request failed unexpectedly")

    def get(self, endpoint: str, params: Mapping[str, JSONScalar] | None = None) -> JSONObject:
        return self._request(method="GET", endpoint=endpoint, params=params)

    def post(self, endpoint: str, json: Mapping[str, JSONValue]) -> JSONObject:
        return self._request(method="POST", endpoint=endpoint, json=json)

    def paginate(
        self,
        endpoint: str,
        params: Mapping[str, JSONScalar] | None = None,
        max_records: int | None = None,
    ) -> Iterator[dict[str, JSONValue]]:
        request_params: dict[str, JSONScalar] = dict(params or {})
        yielded = 0

        while True:
            response = self.get(endpoint=endpoint, params=request_params)
            items_value = response.get("items", [])
            items = cast(object, items_value)
            if not isinstance(items, list):
                raise OWSAPIError("OWS API pagination response contains non-list items field")
            items_list = cast(list[object], items)

            for item in items_list:
                if not isinstance(item, dict):
                    raise OWSAPIError("OWS API pagination item is not an object")
                yield cast(dict[str, JSONValue], item)
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    return

            search_after = response.get("search_after")
            if not search_after:
                return
            if not isinstance(search_after, str):
                raise OWSAPIError("OWS API pagination search_after value must be a string")
            request_params["search_after"] = search_after

    def test_connection(self) -> bool:
        try:
            _ = self.get("/refs/ref_enstr")
            return True
        except OWSAPIError as error:
            LOGGER.error("OWS connection test failed: %s", error)
            return False
