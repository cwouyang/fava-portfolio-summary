#!/usr/bin/env python
"""
Calculate the MWRR and/or TWRR for a list of portfolios

This code is originally from https://github.com/hoostus/portfolio-returns
Copyright Justus Pendleton
It was originally licensed under the Parity Public License 7.0.
This file is dual licensed under the Parity Public License 7.0 and the MIT License
as permitted by the Parity Public License
"""
# pylint: disable=logging-fstring-interpolation broad-except
import argparse
import collections
import datetime
import functools
import logging
import operator
import re
from decimal import Decimal
from pprint import pprint
from typing import Tuple, List

import beancount.core
import beancount.core.convert
import beancount.core.data
import beancount.core.getters
import beancount.loader
import beancount.parser
import beancount.utils
import sys
import time
from dateutil.relativedelta import relativedelta
from fava.helpers import BeancountError

# https://github.com/peliot/XIRR-and-XNPV/blob/master/financial.py
try:
    from scipy.optimize import newton as secant_method  # pylint: disable=import-error
except Exception:
    def secant_method(f, x0, tol=0.0001):
        """
        Solve for x where f(x)=0, given starting x0 and tolerance.
        """
        # pylint: disable=invalid-name
        x1 = x0 * 1.1
        while abs(x1 - x0) / abs(x1) > tol:
            x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        return x1


def xnpv(rate: float, cashflows: List[Tuple[datetime.datetime, float]]) -> float:
    """
    Calculate the net present value of a series of cashflows at irregular intervals.

    :param rate: the discount rate to be applied to the cash flows
    :param cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.
    :return: a single value which is the NPV of the given cash flows.
    :rtype: float

    Notes
    ---------------
    * The Net Present Value is the sum of each of cash flows discounted back to the date of the first cash flow. The
      discounted value of a given cash flow is A/(1+r)**(t-t0), where A is the amount, r is the discount rate, and
      (t-t0) is the time in years from the date of the first cash flow in the series (t0) to the date of the cash flow
      being added to the sum (t).
    * This function is equivalent to the Microsoft Excel function of the same name.
    """
    # pylint: disable=invalid-name
    chron_order = sorted(cashflows, key=lambda x: x[0])
    t0 = chron_order[0][0]  # t0 is the date of the first cash flow

    return sum([cf / (1 + rate) ** ((t - t0).days / 365.0) for (t, cf) in chron_order])


def xirr(cashflows: List[Tuple[datetime.datetime, float]], guess: float = 0.1) -> float:
    """
    Calculate the Internal Rate of Return of a series of cashflows at irregular intervals.

    :param cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.
    :param guess: a guess to be used as a starting point for the numerical solution.
    :return: the IRR as a single value
    :rtype: float

    Notes
    ----------------
    * The Internal Rate of Return (IRR) is the discount rate at which the Net Present Value (NPV) of a series of cash
      flows is equal to zero. The NPV of the series of cash flows is determined using the xnpv function in this module.
      The discount rate at which NPV equals zero is found using the secant method of numerical solution.
    * This function is equivalent to the Microsoft Excel function of the same name.
    * For users that do not have the scipy module installed, there is an alternate version (commented out) that uses
      the secant_method function defined in the module rather than the scipy.optimize module's numerical solver. Both
      use the same method of calculation so there should be no difference in performance, but the secant_method
      function does not fail gracefully in cases where there is no solution, so the scipy.optimize.newton version is
      preferred.
    """
    try:
        return secant_method(lambda r: xnpv(r, cashflows), guess)
    except Exception as _e:
        logging.error("No solution found for IRR: %s", _e)
        return 0.0


def xtwrr(periods, debug=False) -> float:
    """Calculate TWRR from a set of date-ordered periods"""
    dates = sorted(periods.keys())
    last = float(periods[dates[0]][0])
    mean = 1.0
    if debug:
        print("Date          start-balance     cashflow     end-balance     partial")
    for date in dates[1:]:
        cur_bal = float(periods[date][0])
        cashflow = float(periods[date][1])
        partial = 1.0
        # cashflow occurs on end date, so remove it from the current balance
        if last != 0:
            partial = 1 + ((cur_bal - cashflow) - last) / last
        if debug:
            print(f"{date.strftime('%Y-%m-%d')}  {last:-15.2f}  {cashflow:-11.2f}  {cur_bal:-14.2f}  {partial:-10.2f}")
        mean *= partial
        last = cur_bal
    mean = mean - 1.0
    days = (dates[-1] - dates[0]).days
    if days == 0:
        return 0.0
    twrr = (1 + mean) ** (365.0 / days) - 1
    return twrr


def add_position(position, inventory):
    """Add a posting to the inventory"""
    if isinstance(position, beancount.core.data.Posting):
        inventory.add_position(position)
    elif isinstance(position, beancount.core.data.TxnPosting):
        inventory.add_position(position.posting)
    else:
        raise Exception("Not a Posting or TxnPosting", position)


class PostingAccountFilter:
    def __init__(self, patterns=None):
        if patterns:
            self.patterns = re.compile(fr'^(?:{"|".join(patterns)})$')
        else:
            self.patterns = re.compile('^$')
        self.cache = {}

    def is_passed(self, posting):
        account = posting.account
        if account not in self.cache:
            self.cache[account] = bool(self.patterns.search(account))
        return self.cache[account]


class IRR:
    """Wrapper class to allow caching results of multiple calculations to improve performance"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, entries, price_map, currency, errors=None):
        self.all_entries = entries
        self.price_map = price_map
        self.currency = currency
        self.market_value = {}
        self.times = [0, 0]
        self.errors = errors
        # The following reset after each calculate call()
        self.remaining = collections.deque()
        self.inventory = beancount.core.inventory.Inventory()

    def _error(self, msg, meta=None):
        if self.errors:
            if not any(_.source == meta and _.message == msg and _.entry is None for _ in self.errors):
                self.errors.append(BeancountError(meta, msg, None))

    def elapsed(self) -> float:
        """Elapsed time of all runs of calculate()"""
        return sum(self.times)

    def iter_interesting_postings(self, date, posting_account_filter, postings=None):
        """Iterator for 'interesting' postings up-to a specified date"""
        if postings:
            remaining_postings = collections.deque(postings)
        else:
            remaining_postings = self.remaining
        while remaining_postings:
            entry = remaining_postings.popleft()
            if entry.date > date:
                remaining_postings.appendleft(entry)
                break
            for _p in entry.postings:
                if posting_account_filter.is_passed(_p):
                    yield _p

    def get_inventory_as_of_date(self, date, posting_account_filter, postings=None):
        """Get postings up-to a specified date"""
        if postings:
            inventory = beancount.core.inventory.Inventory()
        else:
            inventory = self.inventory
        for _p in self.iter_interesting_postings(date, posting_account_filter, postings):
            add_position(_p, inventory)
        return inventory

    def get_value_as_of(self, date, posting_account_filter, postings=None):
        """Get balance for a list of postings at a specified date"""
        inventory = self.get_inventory_as_of_date(date, posting_account_filter, postings)
        # balance = inventory.reduce(beancount.core.convert.convert_position, self.currency, self.price_map, date)
        balance = beancount.core.inventory.Inventory()
        if date not in self.market_value:
            self.market_value[date] = {}
        date_cache = self.market_value[date]
        for position in inventory:
            value = date_cache.get(position)
            if not value:
                value = beancount.core.convert.convert_position(position, self.currency, self.price_map, date)
                if value.currency != self.currency:
                    # try to convert position via cost
                    if position.cost and position.cost.currency == self.currency:
                        value = beancount.core.amount.Amount(position.cost.number * position.units.number,
                                                             self.currency)
                    else:
                        continue
                date_cache[position] = value
            balance.add_amount(value)
        amount = balance.get_currency_units(self.currency)
        return amount.number

    def calculate(self, patterns, internal_patterns=None, start_date=None, end_date=None,
                  mwr=True, twr=False,
                  cashflows=None, inflow_accounts=None, outflow_accounts=None,
                  debug_twr=False):
        """Calculate MWRR or TWRR for a set of accounts"""
        # pylint: disable=too-many-branches too-many-statements too-many-locals too-many-arguments
        if cashflows is None:
            cashflows = []
        if inflow_accounts is None:
            inflow_accounts = set()
        if outflow_accounts is None:
            outflow_accounts = set()
        if not start_date:
            start_date = datetime.date.min
        if not end_date:
            end_date = datetime.date.today()
        interesting_posting_account_filter = PostingAccountFilter(patterns)
        internal_posting_account_filter = PostingAccountFilter(internal_patterns)

        interesting_txns = self.collect_interesting_txns(interesting_posting_account_filter)
        self.remaining = collections.deque(interesting_txns)
        self.inventory.clear()
        twrr_periods = {}

        # p1 = get_inventory_as_of_date(datetime.date(2000, 3, 31), interesting_txns)
        # p2 = get_inventory_as_of_date(datetime.date(2000, 4, 17), interesting_txns)
        # p1a = get_inventory_as_of_date(datetime.date(2000, 3, 31), None)
        # p2a = get_inventory_as_of_date(datetime.date(2000, 4, 17), None)

        for txns in interesting_txns:
            txns_date = txns.date
            if not start_date <= txns_date <= end_date:
                continue

            # Imagine an entry that looks like
            # [Posting(account=Assets:Brokerage, amount=100),
            #  Posting(account=Income:Dividend, amount=-100)]
            # We want that to net out to $0
            # But an entry like
            # [Posting(account=Assets:Brokerage, amount=100),
            #  Posting(account=Assets:Bank, amount=-100)]
            # should net out to $100
            # we loop over all postings in the entry. if the posting
            # is for an account we care about e.g. Assets:Brokerage then
            # we track the cashflow. But we *also* look for "internal"
            # cashflows and subtract them out. This will leave a net $0
            # if all the cashflows are internal.
            cashflow = self.compute_cacheflow_from_transaction(txns, interesting_posting_account_filter,
                                                               internal_posting_account_filter, inflow_accounts,
                                                               outflow_accounts)
            # calculate net cashflow & the date
            if cashflow.quantize(Decimal('.01')) != 0:
                cashflows.append((txns_date, cashflow))
                if twr:
                    if txns_date not in twrr_periods:
                        twrr_periods[txns_date] = [
                            self.get_value_as_of(txns_date, interesting_posting_account_filter), 0]
                    twrr_periods[txns_date][1] += cashflow

        start_value = self.get_value_as_of(start_date, interesting_posting_account_filter, interesting_txns)
        end_value = self.get_value_as_of(end_date, interesting_posting_account_filter)
        self.adjust_twrr_periods(twrr_periods, start_date, start_value, end_date, end_value)
        self.adjust_cashflows(cashflows, start_date, start_value, end_date, end_value)

        irr = None
        twrr = None
        elapsed = [0, 0, 0]
        elapsed[0] = time.time()
        if mwr:
            if cashflows:
                # we need to coerce everything to a float for xirr to work...
                irr = xirr([(d, float(f)) for (d, f) in cashflows])
            else:
                logging.error(f'No cashflows found during the time period {start_date} -> {end_date}')
        elapsed[1] = time.time()
        if twr and twrr_periods:
            twrr = xtwrr(twrr_periods, debug=debug_twr)
        elapsed[2] = time.time()
        for i in range(2):
            delta = elapsed[i + 1] - elapsed[i]
            self.times[i] += delta
            # print(f"T{i}: delta")
        return irr, twrr

    def collect_interesting_txns(self, posting_account_filter):
        """ Collect transactions that link to any accounts we interest 
        """

        def is_interesting_entry(entry):
            for posting in entry.postings:
                if posting_account_filter.is_passed(posting):
                    return True
            return False

        only_txns = beancount.core.data.filter_txns(self.all_entries)
        interesting_txns = filter(is_interesting_entry, only_txns)
        return list(interesting_txns)

    def compute_cacheflow_from_transaction(self, txns, interesting_posting_account_filter,
                                           internal_posting_account_filter, inflow_accounts,
                                           outflow_accounts) -> Decimal:
        cashflow = Decimal(0)
        for posting in txns.postings:
            try:
                value = self.convert_posting_value_to_target_currency(posting, txns.date, self.currency)
            except ValueError:
                continue

            if interesting_posting_account_filter.is_passed(posting):
                cashflow += value
            elif internal_posting_account_filter.is_passed(posting):
                cashflow += value
            else:
                if value > 0:
                    outflow_accounts.add(posting.account)
                else:
                    inflow_accounts.add(posting.account)
        return cashflow

    def convert_posting_value_to_target_currency(self, posting, date, currency) -> float:
        # convert_position uses the price-map to do price conversions, but this does not necessarily
        # accurately represent the cost at transaction time (due to intra-day variations).  That
        # could cause inaccuracy, but since the cashflow is applied to the daily balance, it is more
        # important to be consistent with values
        converted = beancount.core.convert.convert_position(
            posting, currency, self.price_map, date)
        if converted.currency != currency:
            # If the price_map does not contain a valid price, see if it can be calculated from cost
            # This must align with get_value_as_of()
            if posting.cost and posting.cost.currency == currency:
                value = posting.cost.number * posting.units.number
            else:
                error_msg = f'Could not convert posting {converted} from {date}'
                logging.error(error_msg +
                              f' at {posting.meta["filename"]}:{posting.meta["lineno"]} to {currency}. '
                              'IRR will be wrong.')
                self._error(f"{error_msg}, IRR will be wrong", posting.meta)
                raise ValueError(error_msg)
        else:
            value = converted.number
        return value
    
    def adjust_twrr_periods(self, twrr_periods, start_date, start_value, end_date, end_value):
        if start_date not in twrr_periods and start_date != datetime.date.min:
            twrr_periods[start_date] = [start_value, 0]  # We want the after-cashflow value
        if end_date not in twrr_periods:
            twrr_periods[end_date] = [end_value, 0]

    def adjust_cashflows(self, cashflows, start_date, start_value, end_date, end_value):
        # the start_value will include any cashflows that occurred on that date...
        # this leads to double-counting them, since they'll also appear in our cashflows
        # list. So we need to deduct them from start_value
        opening_txns = [amount for (date, amount) in cashflows if date == start_date]
        start_value -= functools.reduce(operator.add, opening_txns, 0)
        # if starting balance isn't $0 at starting time period then we need a cashflow
        if start_value != 0:
            cashflows.insert(0, (start_date, start_value))
        # if ending balance isn't $0 at end of time period then we need a cashflow
        if end_value != 0:
            cashflows.append((end_date, -end_value))



class ArgsParser:
    """Parser used for parsing arguments when run irr.py directly"""

    def __init__(self):
        self.year = 'year'
        self.ytd = 'ytd'
        one_year = '1year'
        two_year = '2year'
        three_year = '3year'
        five_year = '5year'
        ten_year = '10year'

        parser = argparse.ArgumentParser(
            description="Calculate return data."
        )
        parser.add_argument('bean', help='Path to the beancount file.')
        parser.add_argument('--currency', default='USD', help='Currency to use for calculating returns.')
        parser.add_argument('--account', action='append', default=[],
                            help='Regex pattern of accounts to include when calculating returns. Can be specified multiple times.')
        parser.add_argument('--internal', action='append', default=[],
                            help='Regex pattern of accounts that represent internal cashflows (i.e. dividends or interest)')
        date_format = '%Y-%m-%d'
        parser.add_argument('--from', dest='date_from',
                            type=lambda d: datetime.datetime.strptime(d, date_format).date(),
                            help='Start date: YYYY-MM-DD, 2016-12-31')
        parser.add_argument('--to', dest='date_to', type=lambda d: datetime.datetime.strptime(d, date_format).date(),
                            help='End date YYYY-MM-DD, 2016-12-31')
        date_range = parser.add_mutually_exclusive_group()
        date_range.add_argument(f'--{self.year}', default=False, type=int, help='Year. Shorthand for --from/--to.')
        date_range.add_argument(f'--{self.ytd}', action='store_true')
        date_range.add_argument(f'--{one_year}', action='store_true')
        date_range.add_argument(f'--{two_year}', action='store_true')
        date_range.add_argument(f'--{three_year}', action='store_true')
        date_range.add_argument(f'--{five_year}', action='store_true')
        date_range.add_argument(f'--{ten_year}', action='store_true')
        parser.add_argument('--debug-inflows', action='store_true',
                            help='Print list of all inflow accounts in transactions.')
        parser.add_argument('--debug-outflows', action='store_true',
                            help='Print list of all outflow accounts in transactions.')
        parser.add_argument('--debug-cashflows', action='store_true',
                            help='Print list of all cashflows used for the IRR calculation.')
        parser.add_argument('--debug-twr', action='store_true',
                            help='Print calculations for TWR.')
        self.parser = parser

        self.year_to_offset = {
            one_year: -1,
            two_year: -2,
            three_year: -3,
            five_year: -5,
            ten_year: -10
        }

    def parse(self):
        args = self.parser.parse_args()
        return self._postprocess_args(args)

    def _postprocess_args(self, args):
        shortcuts = [self.year, self.ytd]
        shortcuts.extend(self.year_to_offset.keys())
        shortcut_used = functools.reduce(operator.__or__, [getattr(args, x) for x in shortcuts])
        if shortcut_used and (args.date_from or args.date_to):
            raise Exception('Date shortcut options mutually exclusive with --to/--from options')

        today = datetime.date.today()
        if getattr(args, self.year):
            args.date_from = datetime.date(args.year, 1, 1)
            args.date_to = datetime.date(args.year, 12, 31)
        elif getattr(args, self.ytd):
            args.date_from = datetime.date(today.year, 1, 1)
            args.date_to = today
        else:
            for year, offset in self.year_to_offset.items():
                if getattr(args, year):
                    args.date_from = today + relativedelta(years=offset)
                    args.date_to = today
                    break
        # Set default value when no flag was provided
        if args.date_from is None:
            args.date_from = datetime.date.min
        if args.date_to is None:
            args.date_to = today

        # Clean up attributes that will not be used anymore for following operation
        delattr(args, self.year)
        delattr(args, self.ytd)
        for year in self.year_to_offset.keys():
            delattr(args, year)

        return args


def main():
    """Entrypoint"""
    # pylint: disable=too-many-branches too-many-statements
    logging.basicConfig(format='%(levelname)s: %(message)s')
    args = ArgsParser().parse()

    entries, _errors, _options = beancount.loader.load_file(args.bean, logging.info, log_errors=sys.stderr)
    price_map = beancount.core.prices.build_price_map(entries)

    cashflows = []
    inflow_accounts = set()
    outflow_accounts = set()
    irr, twr = IRR(entries, price_map, args.currency).calculate(
        args.account, internal_patterns=args.internal, start_date=args.date_from, end_date=args.date_to,
        mwr=True, twr=True,
        cashflows=cashflows, inflow_accounts=inflow_accounts, outflow_accounts=outflow_accounts,
        debug_twr=args.debug_twr)
    if irr:
        print(f"IRR: {irr}")
    if twr:
        print(f"TWR: {twr}")
    if args.debug_cashflows:
        print('[cashflows]')
        pprint(cashflows)
    if args.debug_inflows:
        print('>> [inflows]')
        pprint(inflow_accounts)
    if args.debug_outflows:
        print('<< [outflows]')
        pprint(outflow_accounts)


if __name__ == '__main__':
    main()
