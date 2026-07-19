"""
Maps each of Bitext's 27 real intents to:
  1. our 4-category classifier schema (billing / technical / account / general)
  2. a hand-authored summary template (written once per intent, by a human,
     not generated per-example by an LLM)

This is the "ground truth authoring" step. There are only 27 intents, so
each template below was written and reviewed as a single judgment call about
what that category of request means -- the entity values that fill the
{placeholders} come from the real dataset, not from a generative model.
"""

from __future__ import annotations

# intent -> (our_category, summary_template)
# Template placeholders match the entity slot names used in build_golden_dataset.py
INTENT_MAP: dict[str, tuple[str, str]] = {
    # --- ACCOUNT ---
    "create_account": ("account", "Customer wants to create a new account."),
    "delete_account": ("account", "Customer wants to delete their account."),
    "edit_account": ("account", "Customer wants to update their account details."),
    "recover_password": ("account", "Customer is locked out and needs help recovering their password."),
    "registration_problems": ("account", "Customer is having trouble completing account registration."),
    "switch_account": ("account", "Customer wants to switch to a different account type."),

    # --- BILLING ---
    "check_cancellation_fee": ("billing", "Customer wants to know the cancellation fee before cancelling."),
    "check_invoice": ("billing", "Customer has a question about an invoice."),
    "get_invoice": ("billing", "Customer is requesting a copy of an invoice."),
    "check_payment_methods": ("billing", "Customer wants to know what payment methods are accepted."),
    "payment_issue": ("billing", "Customer is reporting a problem with a payment."),
    "check_refund_policy": ("billing", "Customer wants to understand the refund policy."),
    "get_refund": ("billing", "Customer is requesting a refund."),
    "track_refund": ("billing", "Customer wants to check the status of a refund already requested."),

    # --- GENERAL ---
    "contact_customer_service": ("general", "Customer wants to know how to reach customer service."),
    "contact_human_agent": ("general", "Customer is asking to speak with a human agent."),
    "delivery_options": ("general", "Customer is asking about available delivery options."),
    "delivery_period": ("general", "Customer wants to know how long delivery will take."),
    "complaint": ("general", "Customer is filing a complaint about their experience."),
    "review": ("general", "Customer wants to leave a review or feedback."),
    "cancel_order": ("general", "Customer wants to cancel an order."),
    "change_order": ("general", "Customer wants to modify an existing order."),
    "place_order": ("general", "Customer needs help placing an order."),
    "track_order": ("general", "Customer wants to track the status of an order."),
    "change_shipping_address": ("general", "Customer wants to change their shipping address."),
    "set_up_shipping_address": ("general", "Customer wants to set up a shipping address."),
    "newsletter_subscription": ("general", "Customer has a question about newsletter subscription."),
}
