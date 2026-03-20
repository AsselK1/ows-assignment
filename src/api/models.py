from __future__ import annotations

from pydantic import BaseModel, Field


class OWSResponse(BaseModel):
    items: list[dict[str, object]] = Field(default_factory=list)
    total: int = 0
    search_after: str | None = None


class Subject(BaseModel):
    id: int | None = None
    bin: str | None = None
    name_ru: str | None = None
    name_kz: str | None = None


class Plan(BaseModel):
    id: int | None = None
    customer_bin: str | None = None
    year: int | None = None
    enstr_code: str | None = None


class Announcement(BaseModel):
    id: int | None = None
    customer_bin: str | None = None
    total_sum: float | None = None
    ref_buy_status_id: int | None = None


class Lot(BaseModel):
    id: int | None = None
    trd_buy_id: int | None = None
    amount: float | None = None
    enstr_code: str | None = None


class Contract(BaseModel):
    id: int | None = None
    supplier_bin: str | None = None
    customer_bin: str | None = None
    contract_sum: float | None = None


class ContractAct(BaseModel):
    id: int | None = None
    contract_id: int | None = None
    act_sum: float | None = None


class RefENSTR(BaseModel):
    id: int | None = None
    code: str | None = None
    name_ru: str | None = None
    name_kz: str | None = None


class RefKATO(BaseModel):
    id: int | None = None
    code: str | None = None
    name_ru: str | None = None
    name_kz: str | None = None


class RefMKEI(BaseModel):
    id: int | None = None
    code: str | None = None
    name_ru: str | None = None
    name_kz: str | None = None
