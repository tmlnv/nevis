"""Populate a running API with demo data. Usage: uv run python seed.py [base_url]"""

import os
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
TOKEN = os.environ["API_BEARER_TOKEN"]

CLIENTS = [
    {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@neviswealth.com",
        "description": "Long-standing private client, balanced growth mandate.",
        "social_links": ["https://www.linkedin.com/in/johndoe"],
        "documents": [
            {
                "title": "Utility bill",
                "content": (
                    "Electricity utility bill issued by City Power for the billing period "
                    "1 March 2026 to 31 March 2026. Service address: 14 Oak Lane, Bristol, "
                    "BS1 4TR. Account holder: John Doe. Total amount due GBP 84.20, payable "
                    "by 15 April 2026. This document is routinely accepted as evidence of "
                    "residential address for onboarding checks."
                ),
            },
            {
                "title": "Portfolio review Q1 2026",
                "content": (
                    "Quarterly investment portfolio review for the Doe family trust. The "
                    "balanced mandate returned 3.1% over the quarter against a benchmark of "
                    "2.4%. Equity exposure was reduced from 62% to 57% in favour of "
                    "short-duration government bonds following the March rate decision."
                ),
            },
        ],
    },
    {
        "first_name": "Amara",
        "last_name": "Okonkwo",
        "email": "amara.okonkwo@meridiancapital.co.uk",
        "description": "Corporate client, treasury and cash management.",
        "social_links": [],
        "documents": [
            {
                "title": "Signed tenancy agreement",
                "content": (
                    "Assured shorthold tenancy agreement confirming that the tenant resides "
                    "at 42 Cathedral Road, Cardiff, CF11 9LJ, for a fixed term of twelve "
                    "months commencing 1 February 2026. Signed by both landlord and tenant "
                    "and witnessed. Frequently used to establish proof of residence."
                ),
            }
        ],
    },
    {
        "first_name": "Henrik",
        "last_name": "Sørensen",
        "email": "h.sorensen@nordvest-family.dk",
        "description": "Family office contact, Nordic mandates and ESG screening.",
        "social_links": ["https://nordvest-family.dk/team/henrik"],
        "documents": [
            {
                "title": "Passport copy",
                "content": (
                    "Certified colour copy of a Danish passport together with a scan of the "
                    "national identity card, supplied for identity verification during "
                    "client onboarding. Document expires 4 September 2031."
                ),
            },
            {
                "title": "ESG screening policy",
                "content": (
                    "Exclusion policy applied across all Nordvest discretionary mandates. "
                    "Issuers deriving more than 5% of revenue from thermal coal extraction, "
                    "tobacco production, or controversial weapons are excluded outright."
                ),
            },
        ],
    },
]


def main() -> None:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.Client(base_url=BASE, headers=headers, timeout=120) as http:
        for entry in CLIENTS:
            documents = entry.pop("documents")
            response = http.post("/clients", json=entry)
            response.raise_for_status()
            client_id = response.json()["id"]
            print(f"client {entry['email']} -> {client_id}")
            for document in documents:
                created = http.post(f"/clients/{client_id}/documents", json=document)
                created.raise_for_status()
                print(f"  document {document['title']!r} -> {created.json()['id']}")


if __name__ == "__main__":
    main()
