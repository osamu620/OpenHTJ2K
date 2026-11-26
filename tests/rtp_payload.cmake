# RFC 9828 payload headers as the RTP receiver parses them (tests/tools/rfc9828_parser_check):
# a Main Packet with XTRAC != 0 is accepted and its XTRAB skipped (§8.4).
add_test(NAME rfc9828_main_xtrab COMMAND rfc9828_parser_check)
