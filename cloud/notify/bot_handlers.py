"""Backward-compatibility shim. Real implementations live in handlers/.

All handlers were extracted to cloud/notify/handlers/ submodules:
- _shared.py: ADMIN_CHAT_IDS, _registration_state, _haversine, _safe_answer
- commands.py: /status, /stop, /test, /help, /rangers
- registration.py: /start, district_chosen, confirm_reg, /cancel
- incident.py: accept, location, verdict, snooze, dispatch_drone
- evidence.py: voice, photo, protocol generation
- alerts.py: rag_callback
- __init__.py: get_handlers() factory and composite text_handler
"""

from cloud.notify.handlers import get_handlers, text_handler  # noqa: F401
from cloud.notify.handlers._shared import (  # noqa: F401
    ADMIN_CHAT_IDS,
    _haversine,
    _REG_STEP_BADGE,
    _REG_STEP_CONFIRM,
    _REG_STEP_NAME,
    _REG_TTL,
    _registration_state,
    _safe_answer,
)
from cloud.notify.handlers.alerts import rag_callback  # noqa: F401
from cloud.notify.handlers.commands import (  # noqa: F401
    help_cmd,
    rangers_cmd,
    status,
    stop,
    test_alert,
)
from cloud.notify.handlers.evidence import (  # noqa: F401
    _generate_and_send_protocol,
    handle_inspector_photo,
    voice_handler,
)
from cloud.notify.handlers.incident import (  # noqa: F401
    _snooze_resend,
    accept_callback,
    dispatch_drone_callback,
    location_handler,
    snooze_callback,
    verdict_callback,
)
from cloud.notify.handlers.registration import (  # noqa: F401
    cancel_cmd,
    confirm_reg_callback,
    district_chosen,
    start,
)
