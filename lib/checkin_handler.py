from __future__ import annotations

import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from multiprocessing import Lock, Process
from typing import TYPE_CHECKING, Any

from .log import get_logger
from .utils import (
    AirportCheckInError,
    DriverTimeoutError,
    RequestError,
    get_current_time,
    make_request,
)

if TYPE_CHECKING:
    from .checkin_scheduler import CheckInScheduler
    from .flight import Flight

# Type alias for JSON
JSON = dict[str, Any]

CHECKIN_URL = "mobile-air-operations/v1/mobile-air-operations/page/check-in/"
MANUAL_CHECKIN_URL = "https://mobile.southwest.com/check-in"

# Should only be relevant for same day flights
MAX_CHECK_IN_ATTEMPTS = 10

logger = get_logger(__name__)


class CheckInHandler:
    """
    Handles checking in for a single flight.

    Sleeps until the flight's check-in time and then attempts the check in.
    """

    def __init__(self, checkin_scheduler: CheckInScheduler, flight: Flight, lock: Lock) -> None:
        self.checkin_scheduler = checkin_scheduler
        self.flight = flight
        self.lock = lock
        self.pid = None

        self.notification_handler = checkin_scheduler.notification_handler
        self.first_name = checkin_scheduler.reservation_monitor.first_name
        self.last_name = checkin_scheduler.reservation_monitor.last_name

    def schedule_check_in(self) -> None:
        logger.debug("Scheduling check-in for current flight")
        process = Process(target=self._set_check_in)
        process.start()
        self.pid = process.pid

    def stop_check_in(self) -> None:
        """
        Terminate the check-in process by killing its process ID. The process can't
        be directly terminated with process.terminate() as the process object cannot
        be pickled (necessary when using multiprocessing's 'spawn' start method).
        """
        logger.debug("Stopping check-in for current flight")

        try:
            logger.debug("Killing process with PID %d", self.pid)
            os.kill(self.pid, signal.SIGTERM)

            # Wait so zombie (defunct) processes are not created
            logger.debug("Waiting for process with PID %d to be terminated", self.pid)
            os.waitpid(self.pid, 0)
        except (ChildProcessError, PermissionError):
            # Processes are handled differently in Windows
            pass

        logger.debug("Process with PID %d successfully terminated", self.pid)

    def _set_check_in(self) -> None:
        # Check-in is 24 hours before the flight departs
        checkin_time = self.flight.departure_time - timedelta(days=1)

        try:
            self._wait_for_check_in(checkin_time)
            self._check_in()
        except KeyboardInterrupt:
            # This is handled in the Reservation Monitor attached to this Checkin Handler
            pass

    def _wait_for_check_in(self, checkin_time: datetime) -> None:
        current_time = get_current_time()
        if checkin_time <= current_time:
            logger.debug("Check-in time has passed. Going straight to check-in")
            return

        # Refresh headers 30 minutes before to make sure they are valid
        sleep_time = (checkin_time - current_time - timedelta(minutes=30)).total_seconds()

        # Only try to refresh the headers if the check-in is more than thirty minutes away
        if sleep_time > 0:
            logger.debug("Sleeping until thirty minutes before check-in...")
            self._safe_sleep(sleep_time)

            # Lock to ensure multiple checkin handlers aren't refreshing headers
            # at the same time (the webdriver doesn't work well with concurrency)
            logger.debug("Acquiring lock...")
            with self.lock:
                logger.debug("Lock acquired")
                try:
                    self.checkin_scheduler.refresh_headers()
                except DriverTimeoutError:
                    logger.debug("Timeout while refreshing headers before check-in")
                    self.notification_handler.timeout_before_checkin(self.flight)

            logger.debug("Lock released")
            current_time = get_current_time()

        sleep_time = (checkin_time - current_time).total_seconds()
        logger.debug("Sleeping until check-in: %d seconds...", sleep_time)
        time.sleep(sleep_time)

    def _safe_sleep(self, total_sleep_time: float) -> None:
        """
        If the total sleep time is too long, an overflow error could occur.
        Therefore, the script will continuously sleep in two week periods
        to avoid this issue.
        """
        two_weeks = 60 * 60 * 24 * 14
        while total_sleep_time > 0:
            sleep_time = min(total_sleep_time, two_weeks)
            time.sleep(sleep_time)
            total_sleep_time -= sleep_time

    def _check_in(self) -> None:
        """
        Checks into a flight. Will catch any errors that occur during the check-in process.
        """
        print(
            f"Checking in to flight from '{self.flight.departure_airport}' to "
            f"'{self.flight.destination_airport}' for {self.first_name} {self.last_name}\n"
        )  # Don't log as it has sensitive information

        try:
            reservation = self._attempt_check_in()
        except AirportCheckInError:
            logger.debug("Failed to check in. Airport check-in is required")
            self.notification_handler.airport_checkin_required(self.flight)
            return
        except RequestError as err:
            logger.debug("Failed to check in. Error: %s. Exiting", err)
            self.notification_handler.failed_checkin(err, self.flight)
            return

        self.notification_handler.successful_checkin(
            reservation["checkInConfirmationPage"], self.flight
        )

    def _attempt_check_in(self) -> JSON:
        """
        Attempts to check in once for all flights using parallel threads, staggered by 2 seconds.
        Stops all threads instantly as soon as one succeeds.
        """
        logger.debug("Attempting to check in")

        expected_flights = 1
        if self.flight.is_same_day:
            logger.debug("Checking in same-day flight")
            expected_flights = 2

        stop_event = threading.Event()
        max_threads = 5
        futures = {}

        def check_in_with_thread(thread_name: str) -> JSON:
            if stop_event.is_set():
                logger.debug("%s exiting early", thread_name)
                return None

            logger.debug("%s starting check-in attempt", thread_name)
            try:
                reservation = self._check_in_to_flight()
                flights = reservation["checkInConfirmationPage"]["flights"]
                if len(flights) >= expected_flights:
                    logger.debug("%s successfully checked in", thread_name)
                    stop_event.set()  # Signal all threads to stop
                    return reservation
            except RequestError as err:
                logger.debug("%s failed to check in: %s", thread_name, err)

            return None

        with ThreadPoolExecutor(max_threads) as executor:
            start_time = time.time()
            for i in range(max_threads):
                if stop_event.is_set():
                    break  # Skip launching new threads if one succeeded
                thread_name = f"Thread-{i + 1}"
                future = executor.submit(check_in_with_thread, thread_name)
                futures[future] = thread_name

                # Ensure stagger only happens if no thread succeeded yet
                while time.time() - start_time < (2 * (i + 1)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)  # Small check-in intervals to exit quickly

            for future in as_completed(futures):
                reservation = future.result()
                if reservation:
                    logger.debug("Check-in succeeded, canceling remaining threads.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return reservation

        logger.debug("All parallel check-in attempts failed.")
        raise RequestError("All parallel check-in attempts failed.")

    def _check_in_to_flight(self) -> JSON:
        """
        First, initiate a POST request to get the needed check-in information. Subsequently, execute
        another POST request to submit the check in.
        """
        headers = self.checkin_scheduler.headers
        info = {
            "firstName": self.first_name,
            "lastName": self.last_name,
            "passengerSearchToken": "",
            "recordLocator": self.flight.confirmation_number,
        }
        site = CHECKIN_URL + self.flight.confirmation_number

        logger.debug("Making first POST request to check in")
        # Don't randomly sleep during the check-in requests to have them go through more quickly
        response = make_request("POST", site, headers, info, random_sleep=False)

        info = response["checkInViewReservationPage"]["_links"]["checkIn"]
        site = f"mobile-air-operations{info['href']}"

        logger.debug("Making second POST request to check in")
        reservation = make_request("POST", site, headers, info["body"], random_sleep=False)
        return reservation
