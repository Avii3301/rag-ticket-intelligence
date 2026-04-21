"""
Generate a synthetic ITSM ticket dataset (10,000 rows).

Each ticket has a realistic description and matching resolution,
with compositional variation to avoid obvious repetition.
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────

NUM_TICKETS = 10000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "itsm_tickets.csv")

PRIORITY_WEIGHTS = {"Low": 0.25, "Medium": 0.40, "High": 0.20, "Critical": 0.15}

# Priority weights per subcategory — makes priority learnable from description text.
# Subcategories with high blast radius (outages, malware, data loss) skew Critical/High.
# Routine provisioning/requests skew Low/Medium.
PRIORITY_WEIGHTS_BY_SUBCATEGORY = {
    # Critical-heavy
    "Service Outage":    {"Critical": 0.60, "High": 0.30, "Medium": 0.08, "Low": 0.02},
    "Malware":           {"Critical": 0.55, "High": 0.30, "Medium": 0.12, "Low": 0.03},
    "Data Integrity":    {"Critical": 0.50, "High": 0.35, "Medium": 0.12, "Low": 0.03},
    "Ransomware":        {"Critical": 0.80, "High": 0.15, "Medium": 0.04, "Low": 0.01},
    # High-heavy
    "Account Lockout":   {"Critical": 0.15, "High": 0.55, "Medium": 0.25, "Low": 0.05},
    "Suspicious Activity": {"Critical": 0.20, "High": 0.50, "Medium": 0.25, "Low": 0.05},
    "Vulnerability":     {"Critical": 0.25, "High": 0.45, "Medium": 0.25, "Low": 0.05},
    "Phishing":          {"Critical": 0.20, "High": 0.45, "Medium": 0.28, "Low": 0.07},
    "Deployment":        {"Critical": 0.15, "High": 0.45, "Medium": 0.30, "Low": 0.10},
    "Performance":       {"Critical": 0.10, "High": 0.40, "Medium": 0.38, "Low": 0.12},
    "Firewall":          {"Critical": 0.10, "High": 0.40, "Medium": 0.35, "Low": 0.15},
    "Crash":             {"Critical": 0.08, "High": 0.40, "Medium": 0.40, "Low": 0.12},
    # Medium-heavy (default-ish)
    "VPN":               {"Critical": 0.05, "High": 0.25, "Medium": 0.50, "Low": 0.20},
    "DNS":               {"Critical": 0.05, "High": 0.25, "Medium": 0.50, "Low": 0.20},
    "Laptop":            {"Critical": 0.03, "High": 0.20, "Medium": 0.52, "Low": 0.25},
    "Outlook":           {"Critical": 0.03, "High": 0.20, "Medium": 0.52, "Low": 0.25},
    "License":           {"Critical": 0.04, "High": 0.22, "Medium": 0.52, "Low": 0.22},
    "Backup":            {"Critical": 0.05, "High": 0.25, "Medium": 0.48, "Low": 0.22},
    # Low-heavy
    "New Account":       {"Critical": 0.01, "High": 0.09, "Medium": 0.30, "Low": 0.60},
    "Permission Request": {"Critical": 0.01, "High": 0.10, "Medium": 0.34, "Low": 0.55},
    "Password Reset":    {"Critical": 0.02, "High": 0.13, "Medium": 0.40, "Low": 0.45},
    "Update":            {"Critical": 0.02, "High": 0.13, "Medium": 0.45, "Low": 0.40},
    "Policy":            {"Critical": 0.01, "High": 0.09, "Medium": 0.35, "Low": 0.55},
    "Scaling":           {"Critical": 0.02, "High": 0.13, "Medium": 0.45, "Low": 0.40},
    "Storage":           {"Critical": 0.03, "High": 0.17, "Medium": 0.45, "Low": 0.35},
}

# Urgency phrases injected into descriptions — the primary signal for priority prediction.
URGENCY_PHRASES = {
    "Critical": [
        "This is completely blocking production — we are losing revenue every minute.",
        "The entire team is blocked and cannot work. This is a P1 emergency.",
        "Production is down. All users affected. Needs immediate escalation.",
        "This is critical — customer-facing services are unavailable right now.",
        "All work has stopped. This needs to be fixed in the next hour.",
        "We have an SLA breach in 30 minutes if this isn't resolved.",
    ],
    "High": [
        "This is blocking my work completely — I can't proceed without this fixed.",
        "I have a client deliverable due in 2 hours and this is stopping me.",
        "Multiple people on my team are affected and we're all blocked.",
        "This is time-sensitive — please treat as high priority.",
        "I've already missed one meeting because of this. Please help urgently.",
        "This is causing significant disruption to my team's work today.",
    ],
    "Medium": [
        "This is slowing me down but I can work around it for now.",
        "Not blocking me completely but I'd like this fixed today if possible.",
        "This is affecting my productivity but not a total blocker.",
        "Would appreciate a fix when you get a chance — moderately impactful.",
        "Happy to work around this short-term but it needs to be addressed.",
    ],
    "Low": [
        "No rush on this — just flagging it when you have time.",
        "This is a minor inconvenience, not urgent at all.",
        "Whenever you get a chance — low priority for me.",
        "Not urgent. Please address when convenient.",
        "This can wait — just wanted to log it so it doesn't get forgotten.",
        "Low priority — happy to wait until your schedule allows.",
    ],
}

STATUS_WEIGHTS = {"Closed": 0.60, "Resolved": 0.15, "Open": 0.10, "In Progress": 0.10, "Pending": 0.05}

DEPARTMENTS = [
    "IT", "Human Resources", "Finance", "Marketing", "Sales",
    "Engineering", "Operations", "Legal", "Customer Support",
    "Executive", "Research", "Product", "Logistics", "Procurement",
]

PRODUCTS_BY_CATEGORY = {
    "Network": [
        "Cisco AnyConnect", "Palo Alto GlobalProtect", "FortiClient",
        "Cisco Meraki", "Aruba ClearPass", "Juniper SRX", "pfSense",
        "Ubiquiti UniFi", "Fortinet FortiGate", "Cisco ISE",
    ],
    "Hardware": [
        "Dell Latitude 5540", "MacBook Pro M3", "HP EliteBook 840",
        "Lenovo ThinkPad X1", "Dell OptiPlex 7010", "HP ZBook Fury",
        "MacBook Air M2", "Dell Precision 5570", "Surface Pro 9",
        "Lenovo ThinkCentre M90q",
    ],
    "Software": [
        "Microsoft 365", "Google Workspace", "Salesforce", "SAP ERP",
        "Jira", "Confluence", "Adobe Creative Cloud", "Tableau",
        "ServiceNow", "Workday", "AutoCAD", "Visual Studio Code",
        "Slack", "Zoom", "Power BI",
    ],
    "Access/Permissions": [
        "Active Directory", "Okta", "Azure AD", "CyberArk",
        "SailPoint", "Duo Security", "Microsoft Entra ID",
        "Ping Identity", "OneLogin", "JumpCloud",
    ],
    "Email/Communication": [
        "Microsoft Outlook", "Microsoft Teams", "Slack", "Zoom",
        "Google Workspace", "Exchange Online", "Webex",
        "Mimecast", "Proofpoint", "Mailchimp",
    ],
    "Database": [
        "Microsoft SQL Server", "PostgreSQL", "MySQL", "Oracle DB",
        "MongoDB", "Redis", "Amazon RDS", "Azure SQL",
        "Snowflake", "Elasticsearch",
    ],
    "Security": [
        "CrowdStrike Falcon", "Splunk SIEM", "Palo Alto Cortex",
        "Microsoft Defender", "Qualys", "Tenable Nessus",
        "SentinelOne", "Carbon Black", "Darktrace", "Rapid7",
    ],
    "Cloud/Infrastructure": [
        "AWS EC2", "Azure Virtual Machines", "Google Cloud Compute",
        "Kubernetes", "Docker", "Terraform", "Jenkins", "GitHub Actions",
        "Ansible", "VMware vSphere", "Amazon S3", "Azure Blob Storage",
    ],
}

FIRST_NAMES = [
    "James", "Maria", "Robert", "Jennifer", "Michael", "Linda", "David",
    "Sarah", "Carlos", "Priya", "Wei", "Fatima", "Ahmed", "Yuki",
    "Olga", "Emmanuel", "Sofia", "Raj", "Ana", "Dmitri", "Kenji",
    "Aisha", "Lars", "Mei", "Ivan", "Chloe", "Hassan", "Ingrid",
    "Tomás", "Nadia", "Patrick", "Amara", "Chen", "Elena", "Marcus",
    "Leila", "Sven", "Ravi", "Marta", "Kwame",
]

LAST_NAMES = [
    "Smith", "Garcia", "Johnson", "Patel", "Williams", "Chen",
    "Brown", "Kim", "Jones", "Nguyen", "Miller", "Singh",
    "Davis", "Martinez", "Anderson", "Tanaka", "Wilson", "Ali",
    "Taylor", "Müller", "Thomas", "Johansson", "Moore", "Petrov",
    "Jackson", "Santos", "White", "Berg", "Harris", "Okafor",
]

ERROR_CODES = [
    "0x80070005", "0x8024402F", "ERR_CONNECTION_TIMED_OUT",
    "DRIVER_IRQL_NOT_LESS_OR_EQUAL", "0x800F081F", "0xc000021a",
    "ERR_SSL_PROTOCOL_ERROR", "0x80004005", "KERNEL_DATA_INPAGE_ERROR",
    "0x80240034", "ERR_NAME_NOT_RESOLVED", "0x8007045D",
    "STATUS_ACCESS_DENIED", "0x80070057", "NTFS_FILE_SYSTEM",
    "ERR_CERT_AUTHORITY_INVALID", "0x80073701", "0x800706BA",
]

BUILDINGS = ["Building A", "Building B", "Building C", "HQ", "Remote", "East Wing", "West Wing", "Annex"]

# ── Greeting / sign-off / filler fragments ─────────────────────────────────────

OPENERS = [
    "Hi,", "Hello,", "Hey team,", "Hi there,", "Good morning,",
    "Hi IT,", "Hello support,", "Hey,", "", "", "", "", "", "",
]

CLOSINGS = [
    "Thanks.", "Thank you.", "Thanks in advance.", "Appreciate the help.",
    "Please advise.", "Let me know if you need more info.",
    "Regards.", "Cheers.", "", "", "", "", "", "", "",
]

EXTRA_CONTEXT = [
    "This started happening yesterday.",
    "It's been like this since Monday.",
    "I noticed this after the last update.",
    "This has been going on for about a week now.",
    "I first saw this issue this morning.",
    "It was working fine until last Friday.",
    "Other people on my team are seeing the same thing.",
    "This is only happening on my machine.",
    "I'm on a tight deadline so this is really blocking me.",
    "Not super urgent but it's getting annoying.",
    "I'm in {building} if someone needs to come look at it.",
    "I'm working remotely today if that matters.",
    "",  "", "", "", "", "",
]

ATTEMPTS = [
    "I already tried restarting my computer.",
    "I've rebooted twice, no change.",
    "I cleared my browser cache but it didn't help.",
    "I tried logging out and back in.",
    "I checked and my internet connection seems fine.",
    "Restarting the application didn't fix it.",
    "I tried the usual troubleshooting steps.",
    "I asked a coworker and they don't have this problem.",
    "", "", "", "", "", "",
]

# ── Ticket templates by category/subcategory ───────────────────────────────────
# Each entry: (subject, description, resolution)
# Placeholders: {user}, {system}, {error}, {building}, {product}, {timeframe}

TEMPLATES = {
    "Network": {
        "VPN": [
            (
                "VPN connection dropping repeatedly",
                "My VPN keeps disconnecting every 10-15 minutes. I'm using {product} and each time it drops I have to re-enter my credentials. Makes it impossible to stay connected to internal resources.",
                "Identified split-tunnel misconfiguration causing keepalive timeouts. Updated VPN profile settings and pushed new configuration. Connection now stable — monitored for 2 hours with no drops."
            ),
            (
                "Cannot connect to VPN from home",
                "I'm trying to connect to the company VPN from my home network but it just times out. Getting {error} when I try to connect. Was working fine yesterday.",
                "User's ISP was blocking UDP port 4500 needed for VPN. Switched VPN profile to TCP/443 fallback mode. Connection established successfully."
            ),
            (
                "VPN extremely slow",
                "Connected to VPN but everything is incredibly slow. Internal apps take 30+ seconds to load. Speed test outside VPN shows 200mbps but on VPN it drops to about 2mbps.",
                "Rerouted user to nearest VPN gateway (was connecting to wrong region). Latency dropped from 280ms to 35ms. Also cleared stale DNS cache entries that were causing additional delays."
            ),
            (
                "VPN shows connected but can't reach internal sites",
                "VPN says connected and I can see the green icon, but I can't reach any internal sites — SharePoint, Jira, internal wikis all timeout. External sites work fine.",
                "DNS resolution was failing for internal domains. Flushed DNS cache and reconfigured DNS to use internal DNS servers when VPN is active. All internal sites now accessible."
            ),
            (
                "VPN certificate expired",
                "Getting a certificate error when trying to connect to VPN. Message says the security certificate has expired or is not yet valid. Can't connect at all.",
                "Renewed the client VPN certificate and pushed updated certificate bundle to user's machine via MDM. VPN connection restored."
            ),
            (
                "Need VPN access for new employee",
                "{user} just started in {department} and needs VPN access to work remotely. Manager approved remote work arrangement starting next week.",
                "Created VPN account, generated credentials, and enrolled user's device in MDM. Sent setup instructions via email. User confirmed successful VPN connection."
            ),
            (
                "VPN two-factor authentication not working",
                "I can enter my VPN password but when it asks for the 2FA code from my authenticator app, it keeps saying invalid. I've verified the time on my phone is correct.",
                "Re-synced user's TOTP seed in the authentication server. Old seed had drifted. Generated new QR code for user to re-enroll in authenticator app. 2FA now working."
            ),
        ],
        "WiFi": [
            (
                "WiFi keeps dropping in conference room",
                "Every time we have a meeting in conference room {building}, the WiFi drops for everyone. Seems to happen when there's more than 6-7 people connected.",
                "Access point in the conference room was overloaded — max client limit was set too low. Upgraded AP firmware and increased max connections from 10 to 30. Also added a second AP for redundancy."
            ),
            (
                "Can't connect to corporate WiFi",
                "My laptop won't connect to the corporate WiFi network. It sees the network but when I enter my credentials it just spins and eventually fails. Personal hotspot works fine.",
                "User's machine certificate for 802.1X authentication had expired. Renewed the certificate through the internal CA and reconnected. WiFi authentication now succeeds."
            ),
            (
                "WiFi speed very slow in {building}",
                "WiFi in {building} has been unusable for the past few days. Speedtest shows less than 5mbps down when it should be around 100. Multiple people are complaining.",
                "Discovered a rogue access point on the same channel causing interference. Removed the unauthorized device and adjusted channel assignments across the floor. Speeds back to normal (95mbps+)."
            ),
            (
                "Guest WiFi not working",
                "We have clients visiting today and the guest WiFi network isn't showing up at all. They need internet access for a presentation in 2 hours.",
                "Guest SSID broadcast had been accidentally disabled during last night's maintenance window. Re-enabled the SSID and verified guest isolation is still properly configured. Network now visible and functional."
            ),
            (
                "WiFi connected but no internet",
                "My laptop shows connected to WiFi with full signal bars, but no pages load. Other people around me seem fine. I've tried forgetting and reconnecting to the network.",
                "User's device had a static IP configured that conflicted with DHCP range. Reset network adapter to DHCP and renewed lease. Internet access restored."
            ),
            (
                "Eduroam/partner WiFi access needed",
                "I need access to the partner WiFi network for a visiting researcher from {department}. They'll be here for 3 months and need reliable connectivity.",
                "Created partner WiFi account with 90-day expiration. Configured VLAN access to limit network segment to guest resources only. Provided credentials and connection guide to visitor."
            ),
        ],
        "DNS": [
            (
                "Internal sites not resolving",
                "Can't reach any *.internal.company.com addresses. Browser shows ERR_NAME_NOT_RESOLVED. External sites like google.com work fine. Multiple people on the same floor affected.",
                "Internal DNS server had stale zone file after failed zone transfer. Forced zone transfer from primary DNS and restarted the DNS service. Internal resolution working for all affected users."
            ),
            (
                "DNS lookup failures causing app timeouts",
                "Our {product} instance keeps timing out and the logs show DNS lookup failures for the database hostname. Happening intermittently — works fine for an hour then fails for 10 minutes.",
                "Secondary DNS server was intermittently unreachable due to a flapping network interface. Replaced the NIC and updated DNS failover configuration to reduce timeout from 30s to 5s."
            ),
            (
                "New subdomain not resolving",
                "We set up a new internal service at reporting.internal.company.com but nobody can reach it. DNS doesn't seem to know about it. The server is running and reachable by IP.",
                "DNS A record had not been added for the new subdomain. Created the record pointing to the correct IP, waited for propagation, and confirmed resolution from multiple clients."
            ),
            (
                "DNS returning wrong IP for internal service",
                "When I nslookup our {product} server it returns a different IP than what it should be. I think someone changed it but now the app is broken because it's pointing to the old decomissioned server.",
                "Found stale DNS record pointing to decommissioned host. Updated A record with correct IP address and flushed DNS caches on affected machines. Service now resolving correctly."
            ),
            (
                "Slow DNS resolution across office",
                "Everything feels sluggish today. Page loads take 5-10 seconds before they even start. Ran some tests and DNS lookups are taking 3-4 seconds each when they should be instant.",
                "DNS forwarder was pointing to an unresponsive external resolver. Updated forwarder configuration to use reliable public DNS as fallback. Resolution times back under 20ms."
            ),
        ],
        "Slow Connection": [
            (
                "Internet extremely slow for entire floor",
                "The whole 3rd floor is experiencing very slow internet. Downloads are crawling, video calls keep freezing. Started around 10am today. We've got about 40 people affected.",
                "Identified a failing switch on the 3rd floor causing packet loss and retransmissions. Replaced the switch and verified throughput returned to normal across all ports."
            ),
            (
                "Network slow only in the afternoon",
                "Network speed is fine in the morning but after about 1pm it becomes nearly unusable. This pattern has been consistent for the past week. Are we hitting bandwidth limits?",
                "Identified that automated backup jobs were scheduled at 12:30pm and consuming 80% of WAN bandwidth. Rescheduled backups to 11pm and implemented QoS policies to prioritize user traffic."
            ),
            (
                "Specific application very slow over network",
                "{product} is painfully slow — takes 30 seconds to load each page. Every other application works fine. Other users are seeing the same thing.",
                "Application server's network interface was negotiating at 100Mbps instead of 1Gbps due to a faulty cable. Replaced the ethernet cable and confirmed 1Gbps link speed. Application performance back to normal."
            ),
            (
                "File transfers to network drive extremely slow",
                "Copying files to the shared network drive is taking forever. A 50MB file takes over 10 minutes when it used to take seconds. Only affects the S: drive, other mapped drives are fine.",
                "The file server hosting S: drive had a degraded RAID array causing heavy I/O latency. Replaced failed disk, rebuilt array, and performance returned to normal. Transfer speeds back to 100MB/s+."
            ),
            (
                "Latency spikes during video calls",
                "Every Teams/Zoom call I'm on has terrible quality. Audio cuts out, video freezes. My ping to 8.8.8.8 shows spikes up to 800ms every few seconds.",
                "User's ethernet port was configured for half-duplex due to a port configuration error. Changed to auto-negotiate full-duplex. Latency stabilized at 12ms with no more spikes."
            ),
        ],
        "Firewall": [
            (
                "Application blocked by firewall",
                "I need to access {product} but I'm getting a connection refused error. I think the firewall might be blocking it. The vendor says we need to allow outbound traffic on port 8443.",
                "Added firewall rule to allow outbound TCP/8443 to the vendor's IP range. Verified connectivity from user's workstation. Application now connects successfully."
            ),
            (
                "New service needs firewall exception",
                "We're deploying a new {product} instance and need ports 443 and 8080 opened between the app server (10.0.5.20) and the database server (10.0.6.15). Change request CR-4521 approved.",
                "Created firewall rules per CR-4521: allowed TCP/443 and TCP/8080 between specified hosts. Rules tested and verified. Notified requestor that the change is live."
            ),
            (
                "Firewall blocking legitimate traffic",
                "Since the firewall update last night, our CI/CD pipeline is broken. Jenkins can't reach our artifact repository. Getting connection timeouts on port 8081.",
                "Last night's firewall rule update accidentally removed the exception for the artifact repository. Restored the rule and added it to the protected rules list to prevent accidental removal."
            ),
            (
                "Suspicious outbound connections detected",
                "Our monitoring is showing unusual outbound connections from workstation {user}-PC to external IPs on port 6667. Flagging for review — could be malware or unauthorized software.",
                "Investigated the workstation and found an unauthorized IRC client installed by the user. Removed the software, blocked port 6667 at the firewall, and reported to security team for policy follow-up."
            ),
            (
                "Need to whitelist vendor IP range",
                "Our new SaaS vendor needs us to whitelist their IP range (203.0.113.0/24) so their platform can push updates to our integration endpoint. Security team has approved.",
                "Whitelisted the approved IP range on the perimeter firewall for inbound HTTPS traffic to the integration endpoint. Verified with vendor that push notifications are now being received."
            ),
        ],
    },
    "Hardware": {
        "Laptop": [
            (
                "Laptop won't turn on",
                "My {product} won't power on at all. No lights, no fan, nothing happens when I press the power button. Was working fine when I shut it down yesterday.",
                "Performed hard reset by removing battery and holding power button for 30 seconds. Reseated RAM and power connection. Laptop powered on — diagnostics show battery was fully depleted due to BIOS update pending. Charged and verified stable."
            ),
            (
                "Laptop extremely slow, takes 15 min to boot",
                "My laptop has been getting slower and slower over the past month. Now it takes about 15 minutes from pressing power to being able to actually open anything. Task manager shows disk at 100% constantly.",
                "Ran diagnostics — hard drive had significant bad sectors and was failing (SMART status: predicted failure). Replaced HDD with SSD, cloned user data, and reimaged. Boot time now under 30 seconds."
            ),
            (
                "Laptop screen flickering",
                "The screen on my {product} has been flickering on and off. Sometimes it goes completely black for a second then comes back. Getting worse — now happens every few minutes.",
                "Identified faulty display cable connection. Opened laptop, reseated the display ribbon cable, and applied strain relief. Flickering resolved. Monitored for 24 hours with no recurrence."
            ),
            (
                "Need laptop replacement — current one is 5 years old",
                "My laptop is really struggling with daily tasks. It's almost 5 years old and constantly freezing during video calls and when I have more than a few browser tabs open. Manager approved a replacement.",
                "Processed hardware replacement request. Provisioned new {product}, migrated user profile and data from old machine. Old laptop wiped and sent to asset recovery. New laptop deployed and verified with user."
            ),
            (
                "Laptop overheating and shutting down",
                "My laptop keeps overheating — the fans run at full speed constantly and it shuts down randomly, I assume from heat. Lost unsaved work twice this week.",
                "Opened laptop and found significant dust buildup blocking air vents and heat sink. Cleaned thoroughly, replaced thermal paste on CPU/GPU. Temperatures dropped from 95°C to 65°C under load. No more unexpected shutdowns."
            ),
            (
                "Laptop keyboard keys not working",
                "Several keys on my laptop keyboard stopped working — the E, R, and T keys specifically. Makes it really hard to type. Started after a small coffee spill last week that I cleaned up right away.",
                "Inspected keyboard — liquid damage to membrane under affected keys. Replaced keyboard assembly. All keys now functional. Recommended user keep liquids away from device."
            ),
            (
                "Laptop battery draining very fast",
                "My {product} battery used to last 6-7 hours, now it barely makes it to 2 hours. Battery health in settings shows 68%. I've had this laptop for about 2 years.",
                "Battery wear level at 32% confirmed degradation. Replaced battery under warranty. New battery showing 100% health, estimated runtime 7-8 hours. Also optimized power settings."
            ),
        ],
        "Monitor": [
            (
                "External monitor not detected",
                "Plugged my external monitor into my {product} and it's not being detected at all. Tried both HDMI and USB-C ports. The monitor works fine with my personal laptop.",
                "Updated display drivers and installed missing USB-C DisplayPort Alt Mode driver. Monitor immediately detected after driver update. Configured as extended display per user preference."
            ),
            (
                "Monitor showing wrong resolution",
                "My monitor is stuck at 1024x768 and I can't change it to the native 2560x1440. The resolution dropdown doesn't show any higher options. Just got this monitor today.",
                "Installed correct monitor driver and updated GPU driver. Native resolution now available in display settings. Set to 2560x1440 at 60Hz as recommended."
            ),
            (
                "Second monitor keeps going to sleep",
                "My second monitor randomly goes black every 20-30 minutes and I have to unplug and replug the cable to get it back. Primary monitor stays on the whole time.",
                "Faulty DisplayPort cable was causing intermittent signal loss. Replaced cable with a certified DP 1.4 cable. Also disabled display link power saving in power options. Monitor now stays on consistently."
            ),
            (
                "Need additional monitor for dual setup",
                "I'd like to request a second monitor for my workstation. I work with spreadsheets and our {product} dashboard simultaneously and constantly switching between them is killing my productivity. Manager OK'd it.",
                "Approved and deployed 27-inch monitor to user's workstation. Installed monitor arm, connected via DisplayPort, and configured dual display layout. User confirmed setup works for their workflow."
            ),
            (
                "Monitor has dead pixels",
                "There are about 5-6 dead pixels in the center of my monitor — they show up as tiny bright dots that never change. Really distracting when working on documents or presentations.",
                "Confirmed stuck pixels with diagnostic test. Monitor is within warranty. Submitted RMA and replaced with new unit. No dead pixels on replacement — verified with pixel test."
            ),
        ],
        "Printer": [
            (
                "Printer not printing — jobs stuck in queue",
                "Sent several documents to the printer on the 2nd floor but nothing comes out. Print queue shows jobs as 'printing' but they never actually print. Other people having the same issue.",
                "Print spooler service had crashed on the print server. Cleared stuck print jobs, restarted the spooler service, and verified printing from multiple workstations. Queue now processing normally."
            ),
            (
                "Printer printing blank pages",
                "The HP printer near my desk is feeding paper through but every page comes out completely blank. The toner was just replaced last week.",
                "New toner cartridge had shipping seal still attached. Removed the protective seal strip, ran cleaning cycle, and printed test page successfully. User confirmed normal printing resumed."
            ),
            (
                "Can't find network printer",
                "I just moved to a new desk in {building} and I can't find any printers when I try to add one. I need access to the nearest printer for daily reports.",
                "Added user's new subnet to the printer discovery scope. Mapped the nearest network printer (3rd floor copier) to user's workstation. Test print successful."
            ),
            (
                "Printer paper jam won't clear",
                "The printer keeps saying paper jam but I've checked everywhere and can't find any stuck paper. Already opened all the trays and the back panel. Error won't go away.",
                "Found a small torn piece of paper caught in the fuser unit roller, not visible from standard access panels. Removed the fragment with tweezers, cleaned the paper path sensors, and reset the printer. Jam error cleared."
            ),
            (
                "Need to set up scanning to email",
                "I need the copier/scanner in {building} to be able to scan documents directly to my email. Currently I have to scan to USB which is really inconvenient.",
                "Configured scan-to-email on the MFP with user's email address. Set up SMTP relay through our mail server. Tested scanning — document arrived in user's inbox within 30 seconds."
            ),
            (
                "Color printing not working",
                "When I print, everything comes out in black and white even though I have color selected in print settings. Need color for a presentation I'm printing for a client meeting tomorrow.",
                "Print policy was restricting color printing for user's group. Updated group policy to allow color printing. Also replaced low cyan cartridge that would have caused color issues. Test color print successful."
            ),
        ],
        "Peripheral": [
            (
                "Wireless mouse not responding",
                "My wireless mouse stopped working. Replaced batteries, re-plugged the USB receiver into different ports, still nothing. The receiver light doesn't even blink.",
                "USB receiver was defective. Replaced with new mouse and receiver from inventory. Paired successfully. User confirmed mouse is tracking properly."
            ),
            (
                "Docking station not recognizing devices",
                "My docking station stopped recognizing my external monitors and USB devices when I dock my laptop. Was working fine last week. Undocking and redocking doesn't help.",
                "Docking station firmware was outdated and incompatible with recent laptop BIOS update. Updated dock firmware to latest version and power-cycled the dock. All peripherals now recognized correctly."
            ),
            (
                "USB ports not working on desktop",
                "None of the front USB ports on my desktop are working. Can't plug in my USB drive or charge my phone. The back ports still work though.",
                "Front panel USB header cable had come loose from the motherboard. Reseated the internal cable connection. All front USB ports now functional."
            ),
            (
                "Webcam poor quality in calls",
                "My built-in webcam makes me look like a pixelated blob in video calls. People keep asking me to turn my camera off because it's distracting. Can I get an external webcam?",
                "Deployed Logitech C920 external webcam to user. Installed drivers, configured as default video device in Teams/Zoom. Video quality confirmed as significantly improved in test call."
            ),
            (
                "Headset microphone not detected",
                "Plugged in my USB headset but the microphone isn't being detected. Audio plays through the headset fine, but Teams/Zoom can't find the mic. Shows 'No input device found.'",
                "USB headset required specific audio driver not included in Windows default package. Installed manufacturer driver, set headset mic as default recording device in sound settings. Mic now detected and working in all apps."
            ),
        ],
        "Desktop": [
            (
                "Desktop making loud grinding noise",
                "My desktop PC started making a loud grinding noise yesterday. It's coming from inside the case, sounds like something is hitting a fan blade.",
                "Opened case and found a loose cable had dropped into the CPU fan. Rerouted cables and zip-tied them away from fans. Also cleaned significant dust buildup while the case was open. Noise eliminated."
            ),
            (
                "Desktop randomly restarting",
                "My desktop keeps restarting without warning — no blue screen, just suddenly shuts off and comes back on. Happened 4 times today already. Losing work each time.",
                "Event logs showed kernel-power errors indicating unexpected shutdown. Ran PSU diagnostics — power supply was failing under load. Replaced PSU with matching wattage unit. No more unexpected restarts."
            ),
            (
                "Need more RAM — computer too slow with current workload",
                "My {product} only has 8GB RAM and I regularly need to run {product} alongside multiple browser tabs and Excel sheets. System grinds to a halt. Can I get a RAM upgrade?",
                "Approved RAM upgrade request. Installed additional 16GB (2x8GB) matching existing spec. System now has 24GB total. User confirmed significant performance improvement with their typical workload."
            ),
            (
                "Desktop not connecting to network after move",
                "Moved my desktop to a new desk and now it can't connect to the network. Ethernet cable is plugged in but the network icon shows disconnected. Cable works fine with my laptop.",
                "Network port at new desk was not activated in the switch configuration. Enabled the port, assigned correct VLAN, and registered device MAC address. Desktop now on the network with full connectivity."
            ),
        ],
    },
    "Software": {
        "Installation": [
            (
                "Can't install {product} — access denied",
                "Trying to install {product} on my workstation but I get an 'access denied' error. I don't have admin rights. This software was approved by my manager for my project work.",
                "Verified software approval in request system. Elevated user's session via LAPS and installed {product}. Added to the pre-approved software list for self-service deployment going forward."
            ),
            (
                "Software installation stuck at 90%",
                "I've been trying to install {product} and the installer gets to about 90% and then just freezes. Left it for over an hour and no progress. Tried three times now.",
                "Installation was failing due to conflicting background process locking a required DLL. Stopped the conflicting service, cleared partial install files, and reran installation. Completed successfully."
            ),
            (
                "Need {product} installed on new machine",
                "Just got my new laptop and I need {product} installed for my role. I had it on my old machine. License key should still be valid — it's a per-user license.",
                "Deactivated license on old machine, installed {product} on new laptop, and activated with existing license key. Verified application launches and user's settings were migrated."
            ),
            (
                "Installation fails with error {error}",
                "Getting error {error} when trying to install {product}. I've tried running as administrator and disabling antivirus temporarily. Still fails at the same point.",
                "Error caused by missing Visual C++ redistributable dependency. Installed required VC++ runtime package first, then {product} installation completed without errors."
            ),
            (
                "Need bulk software deployment for new team",
                "We have 12 new team members starting in {department} next week. They all need {product} and our standard dev tools installed before day one.",
                "Created deployment package in SCCM with all required software. Staged installations on 12 machines overnight. All installations verified successful. Machines ready for new hires."
            ),
            (
                "Can't install update — insufficient disk space",
                "Trying to update {product} but it says I don't have enough disk space. My C: drive is almost full. Don't know what's taking up all the space.",
                "Ran disk cleanup removing 18GB of temp files, old Windows update cache, and orphaned installer files. Disk now has 35GB free. Software update installed successfully."
            ),
        ],
        "Crash": [
            (
                "{product} crashes on startup",
                "{product} crashes immediately when I try to open it. Get a brief flash of the splash screen then it closes. No error message. Was working fine until this morning.",
                "Corrupted user profile config file was causing crash on load. Renamed the config directory to force regeneration of defaults. Application now starts normally. User re-applied their preferences."
            ),
            (
                "{product} freezes when opening large files",
                "Every time I try to open a file larger than about 50MB in {product}, the application freezes completely. Have to force-close it from task manager. Smaller files work fine.",
                "Application was running in 32-bit mode with limited memory allocation. Switched to 64-bit version and increased JVM heap size to 4GB. Large files now open without freezing."
            ),
            (
                "Application crashing with {error}",
                "Getting {error} crash every time I try to use the export function in {product}. Stack trace points to a null reference. This just started happening after the latest update.",
                "Confirmed bug introduced in latest update. Rolled back to previous version and applied vendor's hotfix patch. Export function now working. Monitoring for next stable release."
            ),
            (
                "Excel crashing when running macros",
                "Excel crashes every time I run our department's reporting macro. Just started today — macro has been working for months without issues. Getting an error about memory.",
                "Recent Windows update changed macro security settings and memory allocation. Reset Excel's macro security to previous level and repaired Office installation. Macros now executing properly."
            ),
            (
                "Browser crashing repeatedly",
                "Chrome keeps crashing multiple times a day. Sometimes individual tabs crash, sometimes the whole browser goes down. Losing work in web apps when it happens.",
                "Found 3 conflicting browser extensions causing memory leaks. Disabled problematic extensions, cleared browser cache (was 4.2GB), and updated Chrome to latest version. Stable since fix."
            ),
        ],
        "License": [
            (
                "{product} license expired",
                "Getting a message that my {product} license has expired. Can't use the software at all — it just shows the license expired dialog. I thought we had an enterprise agreement.",
                "Enterprise license server had a sync issue and wasn't renewing client licenses. Resynced the license server and forced a license refresh on user's machine. Software now activated."
            ),
            (
                "All licenses in use for {product}",
                "Trying to open {product} but getting 'all licenses are currently in use.' We have 25 seats but there's no way 25 people are using it right now. This is blocking my work.",
                "Found 8 stale license sessions from users who hadn't properly closed the application. Released the orphaned sessions and implemented automatic session timeout after 4 hours of inactivity. User now has access."
            ),
            (
                "Need additional {product} licenses",
                "Our team has grown and we need 10 more licenses for {product}. Current allocation is fully used and new team members can't access the software.",
                "Submitted purchase request for 10 additional licenses. Procurement approved. Licenses procured and added to the license pool. All new team members now have access."
            ),
            (
                "License key not accepted",
                "I'm entering the license key for {product} exactly as provided but it keeps saying 'invalid license key.' Triple-checked the key for typos. Copied and pasted it directly.",
                "License key was for a different product edition than what was installed. Provided correct key matching the installed edition. Activation successful."
            ),
            (
                "Software reverted to trial mode",
                "My {product} suddenly reverted to trial mode and says I have 14 days left. I've been using the full version for over a year. All my premium features are locked.",
                "License file was corrupted during a disk error. Restored license file from backup and re-validated with the license server. Full features restored."
            ),
        ],
        "Update": [
            (
                "Windows update stuck for hours",
                "Windows Update has been showing 'Installing updates... 47%' for over 3 hours. Afraid to restart in case it breaks something. Is this normal?",
                "Update was stuck due to a corrupted Windows Update component. Ran Windows Update troubleshooter, cleared the SoftwareDistribution folder, and restarted the update service. Updates completed successfully in 20 minutes."
            ),
            (
                "Update broke {product}",
                "After the latest update to {product}, several features I use daily are completely broken. Reports don't generate and the dashboard shows errors. Need to roll back ASAP.",
                "Rolled back {product} to previous version using system restore point. Verified all features working. Notified vendor about the regression. Will schedule update again when vendor provides fix."
            ),
            (
                "Can't update — error {error}",
                "Trying to update {product} and getting error {error} every time. Tried updating through the app and by downloading the installer from the vendor site. Both fail.",
                "Proxy settings were blocking the update server URL. Added vendor's update domain to the proxy whitelist. Update downloaded and installed successfully."
            ),
            (
                "Automatic updates disrupting work",
                "My computer keeps restarting for updates in the middle of the workday. Lost an hour of unsaved work today when it rebooted during a client presentation. This can't keep happening.",
                "Configured Windows Update active hours to prevent restarts during 7am-7pm. Set updates to download but defer installation to maintenance window (Sundays 2am). No more unexpected restarts during work."
            ),
            (
                "Need to update all machines before compliance deadline",
                "Security audit requires all workstations to be on the latest OS patch by end of month. We have about 50 machines in {department} that still need the update.",
                "Created WSUS deployment group for affected machines. Pushed required patches with forced installation deadline. 48 of 50 machines updated successfully. Remaining 2 needed manual intervention due to disk space — cleaned up and patched."
            ),
        ],
        "Compatibility": [
            (
                "Legacy app not working on Windows 11",
                "Our department relies on {product} which is an older application. Since upgrading to Windows 11, it won't launch at all. We need this app daily — there's no replacement yet.",
                "Configured application to run in Windows 8 compatibility mode with admin privileges. Also installed legacy .NET Framework 3.5 component. Application now launches and functions correctly."
            ),
            (
                "Plugin not compatible with latest browser version",
                "The {product} browser plugin stopped working after the latest Chrome update. We need this plugin for our daily workflow — it's how we interact with the client portal.",
                "Plugin vendor hasn't updated for latest Chrome. Installed the extension in developer mode with manifest V3 compatibility shim as temporary fix. Also contacted vendor — update expected in 2 weeks."
            ),
            (
                "New software conflicts with existing tool",
                "After installing {product}, our existing CRM tool started crashing. They seem to conflict somehow. I need both applications for my work.",
                "Both applications were competing for the same local port (8080). Reconfigured {product} to use port 8180 instead. Both applications now run simultaneously without conflicts."
            ),
            (
                "Java version conflict between applications",
                "App A needs Java 8 and App B needs Java 17. Currently have Java 17 installed and App A won't work. Is there a way to have both?",
                "Installed both Java 8 and Java 17 side by side. Configured App A's shortcut to use Java 8 runtime via JAVA_HOME override. App B continues using system default Java 17. Both apps now functional."
            ),
        ],
    },
    "Access/Permissions": {
        "Account Lockout": [
            (
                "Account locked out",
                "I'm locked out of my account. Tried logging in several times and now it says my account has been locked. I have a meeting in 30 minutes and need access urgently.",
                "Unlocked the user's Active Directory account and reset the failed login counter. Account was locked due to incorrect password attempts. User logged in successfully."
            ),
            (
                "Keep getting locked out every morning",
                "Every single morning when I come in, my account is locked out. I change my password, it works for a day, then locked again the next morning. This has been going on for a week.",
                "Found a stale credential stored in Windows Credential Manager from an old mapped drive that was repeatedly trying to authenticate with the old password. Removed the cached credential. Lockouts stopped."
            ),
            (
                "Account locked after password change",
                "Changed my password yesterday and now I'm locked out this morning. I'm sure I'm using the new password. My phone and tablet might still have the old password saved.",
                "Account lockout was caused by mobile devices trying to sync email with old password. Unlocked account, helped user update password on all devices (phone, tablet, home PC). No further lockouts."
            ),
            (
                "Service account keeps locking out",
                "The service account svc-reporting keeps getting locked out causing our automated reports to fail. It's locking out every 2-3 hours. No one is manually using this account.",
                "Traced lockout source to a decommissioned server still running scheduled tasks with this service account. Disabled the tasks on the old server and reset the service account password on all legitimate hosts. Lockouts resolved."
            ),
            (
                "Multiple users locked out simultaneously",
                "About 15 people in {department} all got locked out of their accounts at the same time around 9am. Seems too coordinated to be a coincidence. Are we under attack?",
                "Investigation revealed a misconfigured application in {department} that was attempting authentication with a cached expired token, triggering lockout policy for all connected users. Fixed the application config and unlocked all affected accounts. No security incident."
            ),
        ],
        "Password Reset": [
            (
                "Forgot my password",
                "I forgot my password over the long weekend and now I can't log in. Already tried the self-service reset but it says my security questions don't match. Can someone reset it for me?",
                "Verified user identity through manager confirmation and employee ID. Reset password and set it to force change on next login. Also updated security questions for future self-service resets."
            ),
            (
                "Password expired while on vacation",
                "Was on vacation for 2 weeks and my password expired while I was away. Can't log in to reset it because the old password doesn't work and the new one hasn't been set.",
                "Reset password via admin console. Extended password expiration policy for users on extended leave to prevent this issue. User logged in and set new permanent password."
            ),
            (
                "Self-service password reset not sending email",
                "Tried to reset my password through the self-service portal but the reset email never arrives. Checked spam folder too. Waited 30 minutes, tried 3 times.",
                "SMTP relay for the self-service portal had reached its daily sending limit. Increased the limit and manually triggered the reset email. User received it and successfully reset password."
            ),
            (
                "Need to reset password for shared account",
                "The shared account for the {department} reception desk needs a password reset. No one remembers the current password and we have a new receptionist starting today.",
                "Reset shared account password and communicated new credentials securely to department manager. Also enabled shared mailbox access to reduce reliance on shared login credentials."
            ),
            (
                "Password meets complexity but keeps getting rejected",
                "I'm trying to set a new password that has uppercase, lowercase, numbers, and special characters but it keeps saying it doesn't meet requirements. What am I missing?",
                "Password policy requires that new password cannot contain any part of the username and cannot match any of the last 12 passwords. User's attempted passwords contained their last name. Advised on requirements and user set compliant password."
            ),
        ],
        "Permission Request": [
            (
                "Need access to shared drive",
                "I need read/write access to the \\\\fileserver\\{department} shared drive. I was recently transferred to {department} and my manager {user} has approved the access.",
                "Verified approval with manager. Added user to the {department}-FileShare security group in AD. Access to shared drive confirmed within 15 minutes after group policy refresh."
            ),
            (
                "Need admin access to {product}",
                "I need administrator access to {product} for my new role as team lead. I need to be able to manage user accounts and configure project settings. Manager approved.",
                "Added user to {product} admin role after verifying approval chain. User now has administrative access. Provided brief walkthrough of admin panel capabilities."
            ),
            (
                "Remove access for departed employee",
                "{user} left the company last Friday. Please disable their account and remove all access immediately. They had access to {product} and the {department} shared drive.",
                "Disabled AD account, revoked all application access, removed from all security groups, and forwarded email to manager for 30 days. Followed offboarding checklist. All access confirmed removed."
            ),
            (
                "Need temporary elevated access for project",
                "I need temporary admin access to the staging server for a deployment happening this Thursday. Only need it for about 4 hours. Project manager {user} approved.",
                "Granted temporary admin access via privileged access management (PAM) system with automatic expiration set for 6 hours from deployment start time. Access logged and auditable. Reminded user access will auto-revoke."
            ),
            (
                "Can't access {product} after role change",
                "I changed roles from {department} to Engineering last week but I still can't access the engineering tools. HR says they updated my role in the system.",
                "HR had updated the role but the provisioning sync hadn't run. Manually triggered the identity sync, which updated group memberships. User now has access to all engineering resources. Removed old department access."
            ),
        ],
        "New Account": [
            (
                "New employee needs account setup",
                "We have a new hire starting in {department} on Monday — {user}. They need a full account setup: email, VPN, {product} access, and building badge.",
                "Created AD account, provisioned email and calendar, set up VPN access, granted {product} license, and submitted badge request to facilities. Welcome email with credentials sent to manager for distribution on start date."
            ),
            (
                "Need contractor account",
                "We have a contractor starting a 6-month engagement with {department}. They need limited access — email, {product}, and the {department} shared drive only. No VPN needed as they'll be onsite.",
                "Created contractor account with 6-month expiration. Provisioned email (external-flagged), {product} access, and shared drive. Applied contractor security policy (restricted USB, no admin rights). Credentials sent to hiring manager."
            ),
            (
                "Intern accounts needed for summer program",
                "We have 8 summer interns starting June 1st across {department}. They each need basic accounts with email, internet access, and {product}. Here's the list of names.",
                "Batch-created 8 intern accounts with September 1st expiration. Provisioned email and basic application access for all. Generated credential sheets for each intern. Sent to HR coordinator for distribution."
            ),
            (
                "Account creation for new service",
                "We're deploying a new internal service that needs a service account to run automated tasks. It needs access to the reporting database and the file share.",
                "Created managed service account (gMSA) with principle of least privilege. Granted read-only access to reporting database and read/write to the designated file share path only. Documented service account in CMDB."
            ),
        ],
        "MFA": [
            (
                "MFA not working — can't get verification code",
                "I'm not receiving the MFA verification codes on my phone. Tried SMS and the authenticator app — neither is sending codes. Can't log into anything.",
                "User's MFA enrollment had been corrupted during a backend migration. Re-enrolled user in MFA with new QR code for authenticator app. Also verified phone number for SMS backup. Both methods now working."
            ),
            (
                "Lost phone — can't do MFA",
                "I lost my phone and can't do MFA to log in. I'm completely locked out of everything — email, VPN, all our apps. Need temporary access urgently.",
                "Verified identity through in-person ID check and manager confirmation. Issued temporary bypass code valid for 24 hours. User will re-enroll MFA when new phone is set up."
            ),
            (
                "New phone — need to transfer MFA",
                "Got a new phone and need to transfer my MFA authenticator to it. Old phone has been wiped already so I can't scan the QR code from the old app.",
                "Reset MFA enrollment for user's account. Generated new QR code and helped user set up authenticator on new phone. Verified 3 consecutive codes worked correctly."
            ),
            (
                "MFA prompt appearing too frequently",
                "I'm getting asked for MFA verification every single time I open any app — multiple times per hour. I thought it was supposed to remember my device for 30 days?",
                "User's device was not being recognized due to a browser privacy setting clearing cookies on close. Adjusted trusted device policy and added user's workstation to recognized device list. MFA now only prompts once per 30 days."
            ),
            (
                "Need to enable MFA for entire department",
                "Per the new security policy, all of {department} needs to have MFA enabled by end of this week. About 35 users total. Can you do a bulk enrollment?",
                "Enabled MFA requirement for {department} security group (35 users). Sent enrollment instructions via email to all affected users with a 72-hour grace period. Set up drop-in support hours for users needing help. 33 of 35 enrolled within 48 hours, remaining 2 assisted individually."
            ),
        ],
    },
    "Email/Communication": {
        "Outlook": [
            (
                "Outlook won't open — stuck on loading profile",
                "Outlook gets stuck on 'Loading Profile' and never opens. Waited 20 minutes. Tried restarting my computer twice. I need my email for work.",
                "Outlook profile was corrupted. Created new Outlook profile, reconfigured email account via autodiscover, and migrated user's custom rules and signatures. Outlook now opens in under 10 seconds."
            ),
            (
                "Outlook not syncing — emails are hours behind",
                "My Outlook hasn't synced in about 4 hours. I can see new emails on my phone but they're not showing up on my desktop. Tried send/receive manually but nothing happens.",
                "OST file had grown to 45GB and exceeded sync threshold. Reduced mailbox size by archiving old items, then recreated the OST file. Sync restored and running within normal parameters."
            ),
            (
                "Outlook search not finding emails",
                "Outlook search is completely broken — I search for emails I know exist and it returns zero results. Already tried rebuilding the index from control panel but it didn't help.",
                "Windows Search index for Outlook was corrupted. Deleted the existing index, configured advanced indexing options for Outlook data files, and rebuilt from scratch. Search fully functional after reindex completed (took about 2 hours)."
            ),
            (
                "Can't send emails — getting bounce back",
                "Every email I try to send bounces back with an NDR saying 'message rejected by relay.' Only started today — was sending fine yesterday.",
                "User's sending limit had been triggered by an automated report that sent 500+ emails. Reset the sending quota and helped user configure the report to use a distribution list instead of individual addresses."
            ),
            (
                "Outlook keeps asking for password",
                "Every 10 minutes Outlook pops up asking for my password. I enter it correctly and it goes away but comes back. It's constant and really disruptive.",
                "Cached credentials in Windows Credential Manager were stale. Cleared all Office-related entries from Credential Manager and re-authenticated. Also enabled modern authentication. No more password prompts."
            ),
            (
                "Need to set up out-of-office auto-reply",
                "Going on leave starting tomorrow and need help setting up my out-of-office reply. I've never done it before and can't find the option.",
                "Walked user through setting up automatic replies in Outlook: File > Automatic Replies. Configured internal and external messages with dates. Also set up a forwarding rule to their backup for urgent matters."
            ),
        ],
        "Teams": [
            (
                "Teams calls dropping constantly",
                "Every Teams call I'm on drops after about 5-10 minutes. Audio cuts out first, then I get disconnected. Happens in both meetings and 1:1 calls. Using the desktop app.",
                "Network quality diagnostics showed packet loss on user's connection. Updated network adapter driver and disabled hardware offloading that was causing conflicts. Also switched Teams to use TCP instead of UDP. Calls now stable."
            ),
            (
                "Can't share screen in Teams",
                "When I try to share my screen in Teams meetings, nothing happens. The sharing toolbar appears but my screen doesn't show up for others. They just see a black screen.",
                "GPU hardware acceleration in Teams was conflicting with display driver. Disabled hardware acceleration in Teams settings and updated display driver. Screen sharing now works correctly."
            ),
            (
                "Teams notifications not working",
                "I'm not getting any Teams notifications — no sounds, no pop-ups, nothing. People have been messaging me and I don't see it until I manually open Teams. Missing important messages.",
                "Windows Focus Assist was enabled and suppressing Teams notifications. Disabled Focus Assist and re-enabled Teams notification settings (which had been turned off). Also re-registered Teams in Windows notification system."
            ),
            (
                "Can't join Teams meeting — error",
                "Clicking on Teams meeting links gives me an error saying 'something went wrong.' Can't join any meetings. Tried both the app and the browser — same error.",
                "Teams cache was corrupted. Cleared Teams cache folder (%appdata%/Microsoft/Teams), restarted the application, and re-signed in. Meeting join now works from both app and browser."
            ),
            (
                "Need to create a new Teams channel for project",
                "Need a private Teams channel created for Project Phoenix. Should include about 20 members from {department} and Engineering. Need file sharing and a wiki tab.",
                "Created private channel 'Project Phoenix' in the department Team. Added all 20 members, set up file sharing with proper folder structure, added wiki tab with project template. Channel owners set as project leads."
            ),
        ],
        "Calendar": [
            (
                "Calendar showing wrong time zone",
                "All my calendar events are showing up 3 hours off. A meeting at 10am shows at 1pm on my calendar. Just got back from a business trip and it hasn't corrected itself.",
                "Outlook time zone was still set to the travel destination. Reset time zone to local in Outlook settings and Windows system settings. Calendar events now display at correct times."
            ),
            (
                "Can't see shared calendar",
                "My manager shared their calendar with me last week but I still can't see it in Outlook. They confirmed they shared it and I accepted the sharing invitation.",
                "Calendar permissions were granted but Outlook wasn't showing the shared calendar in the navigation pane. Manually added the shared calendar via Open Calendar > From Address Book. Calendar now visible with correct permissions."
            ),
            (
                "Meeting room double booked",
                "Conference room {building} shows as available when I booked it, but when I went to use it someone else was already in there with their own booking. Calendar system seems broken.",
                "Room mailbox had a permissions issue allowing overlapping bookings. Fixed the resource booking policy to auto-decline conflicting reservations. Resolved the double-booking with both parties and confirmed future conflict detection works."
            ),
            (
                "Calendar invites not being received",
                "People tell me they've sent me meeting invites but they never show up in my calendar or inbox. Checked junk folder too — nothing. I've missed 3 meetings this week because of this.",
                "Mail flow rule was incorrectly routing calendar invites (.ics attachments) to quarantine. Adjusted the mail flow rule to exclude calendar invitation content types. Invites now arriving normally."
            ),
        ],
        "Distribution List": [
            (
                "Need new distribution list created",
                "Need a distribution list created for the new project team — about 15 people from {department}. List name should be 'project-alpha-team'. I have the full member list.",
                "Created distribution list 'project-alpha-team@company.com' in Exchange. Added all 15 members. Set requestor as list owner for self-service management. Test email to list delivered to all members."
            ),
            (
                "Not receiving emails sent to distribution list",
                "I was added to the {department}-all distribution list last week but I'm not receiving any of the group emails. Checked spam/junk — nothing there either.",
                "User was added to the group but their mailbox had a full inbox rule that was routing group emails to a folder they weren't checking. Also confirmed DL membership was active. Adjusted rule and confirmed delivery."
            ),
            (
                "Remove ex-employee from distribution lists",
                "{user} left the company 2 weeks ago but they're still on several distribution lists. Getting bounce-back errors every time we email the team list.",
                "Removed the former employee from all distribution lists (found membership in 7 lists). Cleaned up bounce-back queue. No more NDR errors on group emails."
            ),
            (
                "Distribution list emails going to spam",
                "Emails sent to our {department} distribution list are landing in everyone's spam folder. These are legitimate internal emails but the spam filter flags them.",
                "Distribution list was missing proper SPF and internal sender authentication headers. Updated the list's send-as permissions and added it to the safe senders list in the spam filter policy. Emails now deliver to inbox."
            ),
        ],
    },
    "Database": {
        "Performance": [
            (
                "Database queries extremely slow",
                "Our {product} database queries that used to take seconds are now taking 3-5 minutes. Reports are timing out. The database hasn't changed — same queries, same data volume approximately.",
                "Database statistics were severely outdated — hadn't been refreshed in 3 months. Rebuilt indexes on key tables and updated statistics. Query execution times dropped from minutes back to seconds."
            ),
            (
                "Database CPU at 100% during business hours",
                "The production database server is pegged at 100% CPU from about 9am to 5pm. Everything slows to a crawl. Only happens on weekdays.",
                "Identified a poorly optimized report query running every 15 minutes that was doing full table scans. Added appropriate indexes and rewrote the query to use them. CPU usage dropped from 100% to 35% during peak hours."
            ),
            (
                "Connection pool exhausted — app can't connect to DB",
                "Our application is throwing 'connection pool exhausted' errors. Users are getting timeout errors. We haven't changed any code recently.",
                "Found connection leak in recent application deployment — connections weren't being properly returned to the pool. Fixed the connection disposal in the app code and increased pool size from 50 to 100 as interim measure. Error resolved."
            ),
            (
                "Deadlocks occurring frequently in production",
                "We're seeing frequent deadlock errors in our production database — about 20-30 per hour. Users are seeing random failures when saving data.",
                "Analyzed deadlock graphs and found two stored procedures acquiring locks in opposite order. Refactored both procedures to lock tables in consistent order. Deadlocks eliminated."
            ),
            (
                "Database storage almost full",
                "Production database server is at 92% disk usage. Growing about 2GB per day. If we don't do something we'll run out of space in about 2 weeks.",
                "Identified 180GB of orphaned temp tables and expired audit logs eligible for cleanup. Purged old data, compressed remaining tables, and implemented automated cleanup job. Storage dropped to 64%. Also requested additional storage for long-term growth."
            ),
        ],
        "Backup": [
            (
                "Database backup failing every night",
                "Our nightly database backup has been failing for the past 3 days. Getting error 'insufficient space on backup destination.' Nobody noticed until I checked this morning.",
                "Backup destination drive was full with old backups that weren't being rotated. Implemented retention policy to keep 30 days of backups. Cleared old files and re-ran backup successfully. Added monitoring alert for future failures."
            ),
            (
                "Need to restore database from backup",
                "We need to restore the reporting database from last night's backup. A bad ETL job corrupted several tables with incorrect data this morning.",
                "Restored reporting database from last night's 2am backup to a staging server first to verify integrity. Confirmed data is clean. Swapped in the restored database and re-ran today's ETL with the fixed job. Data now correct."
            ),
            (
                "Backup taking too long — overlaps with business hours",
                "The full database backup now takes 6 hours and is still running when people start work at 8am. It slows everything down. Used to finish by 5am.",
                "Database growth had outpaced backup configuration. Switched from full nightly backup to full weekly + differential nightly. Nightly backup now completes in 45 minutes. Weekly full runs Saturday night."
            ),
            (
                "Backup verification needed for compliance audit",
                "Our compliance team needs verification that database backups are working and restorable. Audit is next week. Can we do a test restore?",
                "Performed full test restore of production database to isolated environment. Verified data integrity with checksums and row counts matching production. Documented results with timestamps for compliance audit."
            ),
        ],
        "Access": [
            (
                "Need read access to production database",
                "I need read-only access to the production {product} database for analytics work. My manager {user} has approved. I only need SELECT permissions on the reporting schema.",
                "Created database login with read-only role on reporting schema only. No access to PII tables per data classification policy. Tested access — user can run queries against approved tables."
            ),
            (
                "Database login failing with authentication error",
                "I can't connect to the database anymore — getting 'Login failed for user' error. My password hasn't changed and I was able to connect yesterday.",
                "User's database account had expired per the 90-day rotation policy. Reset database password and reminded user of the rotation schedule. Also set up a reminder notification 7 days before next expiration."
            ),
            (
                "Need to revoke database access for former team member",
                "{user} transferred out of the analytics team. Please remove their access to the data warehouse. They should no longer have direct database access.",
                "Revoked all database permissions and removed user from analytics DB access group. Dropped their individual database login. Verified no active sessions remain. Documented access change in audit log."
            ),
            (
                "Application can't authenticate to new database server",
                "We migrated to a new database server last weekend and now our {product} application can't connect. Getting authentication errors even though we updated the connection string.",
                "New database server was using Windows Authentication but the application service account wasn't registered on the new host. Added the service account to the new server and granted appropriate roles. Application now connecting."
            ),
        ],
        "Data Integrity": [
            (
                "Duplicate records appearing in database",
                "We're seeing duplicate customer records in the database — some customers appear 3-4 times with slightly different data. It's messing up our reports and billing.",
                "Found missing unique constraint on the import table allowing duplicate inserts. De-duplicated existing records (merged 3,400 duplicates), added unique constraint on customer ID + email, and fixed the import script validation."
            ),
            (
                "Data mismatch between systems",
                "The numbers in {product} don't match what's in the database when I query directly. Revenue report shows $2.3M in the app but $2.1M when I run the same query in SQL.",
                "Application was using a cached materialized view that hadn't been refreshed since last Tuesday. Set up automated view refresh every 4 hours. Numbers now match between app and direct query."
            ),
            (
                "Foreign key violations blocking data load",
                "Our nightly data load is failing with foreign key constraint violations. About 200 records in the staging table reference customer IDs that don't exist in the customer table.",
                "Source system was creating orders before customer records were synced. Reordered the ETL process to load customer data first, then orders. Also added pre-validation step to catch orphaned references before load."
            ),
            (
                "NULL values in required fields",
                "Found a bunch of NULL values in fields that should be mandatory — like order amount and customer name. About 500 records affected. Data was loaded by the automated process.",
                "Input validation was missing in the data pipeline for these fields. Added NOT NULL constraints to the database columns and added validation checks in the ETL pipeline with proper error handling. Manually fixed the 500 affected records from source data."
            ),
        ],
    },
    "Security": {
        "Phishing": [
            (
                "Received suspicious email — possible phishing",
                "I got an email claiming to be from IT asking me to verify my credentials by clicking a link. The sender address looks slightly off — it's 'IT-support@company.co' instead of our actual domain. Didn't click anything.",
                "Confirmed phishing attempt. Blocked sender domain at the email gateway. Scanned all mailboxes and found 47 recipients — removed the phishing email from all inboxes. Sent awareness notification to affected users. No credentials were compromised."
            ),
            (
                "Clicked on phishing link — need help",
                "I accidentally clicked on a link in a suspicious email before I realized it was fake. It asked for my password and I entered it before the page looked wrong and I closed it. What should I do?",
                "Immediately reset user's password and revoked all active sessions. Enabled enhanced monitoring on the account for 30 days. Scanned workstation for malware — clean. Blocked the phishing URL at the firewall. No unauthorized access detected in audit logs."
            ),
            (
                "Phishing email impersonating CEO",
                "Got an email that looks like it's from our CEO asking me to urgently purchase gift cards and send the codes. The email address is slightly different from the real one. Pretty convincing otherwise.",
                "Classic CEO impersonation/BEC attack. Blocked sender, added warning banner for external emails impersonating executive names. Reported to anti-fraud team. Sent company-wide alert about this specific attack pattern."
            ),
            (
                "Multiple people reporting same suspicious email",
                "At least 5 people in {department} have reported getting the same suspicious email about a 'mandatory security update.' None of us clicked it but wanted to report it.",
                "Identified phishing campaign targeting {department}. Quarantined the email across all mailboxes (found in 23 inboxes). Blocked sender IP range. Updated email filter rules to catch similar patterns. Commended users for reporting."
            ),
            (
                "Suspicious attachment in email",
                "Received an email with a .zip attachment from an unknown sender claiming to be an invoice. The email has our company logo but something feels off. Haven't opened the attachment.",
                "Submitted attachment to sandbox for analysis — contained a macro-enabled document with credential-stealing malware. Blocked sender and attachment hash across email gateway. Scanned for any users who may have opened it — none found."
            ),
        ],
        "Malware": [
            (
                "Antivirus detected malware on my PC",
                "CrowdStrike just popped up saying it detected and quarantined a threat on my machine. File was something called 'update_helper.exe'. I don't know where it came from.",
                "Reviewed CrowdStrike detection — file was a trojan downloader. Quarantine was successful, no payload was executed. Ran full system scan — clean. Traced source to a download from an unauthorized software site. Reminded user of software download policy."
            ),
            (
                "Computer behaving strangely — possible malware",
                "My computer has been acting weird — random pop-ups, browser redirecting to strange sites, and it's much slower than usual. Worried I might have malware.",
                "Full scan detected adware/PUP infection (3 items). Removed all detected threats, cleaned browser extensions, reset browser settings, and cleared temp directories. System performance back to normal. Updated endpoint protection signatures."
            ),
            (
                "Ransomware warning on screen",
                "There's a scary message on my screen saying my files are encrypted and I need to pay bitcoin to get them back. I can't open any of my documents. Help!",
                "CRITICAL: Immediately isolated workstation from network. Confirmed ransomware infection contained to local drive only — network shares not affected. Wiped and reimaged machine. Restored user files from last night's backup. Incident reported to security team for full investigation."
            ),
            (
                "USB drive triggered security alert",
                "Plugged in a USB drive I got at a conference and my antivirus immediately flagged something. I ejected the drive right away.",
                "USB contained autorun malware that was caught by endpoint protection before execution. Quarantined threat, scanned system — clean. Confiscated USB drive for analysis. Reminded user about policy against using unknown USB devices."
            ),
        ],
        "Suspicious Activity": [
            (
                "Unusual login from unknown location",
                "Got an alert that someone logged into my account from a location I've never been to. I haven't traveled recently and I definitely didn't log in from there.",
                "Investigated login — originated from a VPN exit node, not user's actual location. Reset user's password as precaution, enabled geo-blocking for high-risk regions, and added conditional access policy requiring MFA for unfamiliar locations. No data access occurred during the suspicious session."
            ),
            (
                "Someone is sending emails from my account",
                "People are telling me they're getting strange emails from me that I never sent. Links to weird websites. My outbox doesn't show them but people are definitely receiving them.",
                "Account was compromised — attacker set up inbox rule to hide sent items. Revoked all sessions, reset password, removed malicious inbox rules, and enabled MFA. Sent retraction notice to all recipients. Audit log shows no data exfiltration."
            ),
            (
                "Unauthorized access attempts in logs",
                "We're seeing hundreds of failed login attempts against our {product} admin portal. All from different IP addresses. Started about 2 hours ago.",
                "Credential stuffing attack in progress. Enabled rate limiting and CAPTCHA on the admin portal. Blocked the attacking IP ranges at the WAF. Confirmed no successful logins from the attack IPs. Rotated admin passwords as precaution."
            ),
            (
                "Employee accessing files outside their scope",
                "Our DLP system flagged a user in {department} accessing sensitive financial documents they shouldn't have access to. They've downloaded about 50 files in the past hour.",
                "Immediately revoked user's access to the financial share. Reviewed access logs — user had been inadvertently granted access during a recent AD group change. Removed incorrect group membership. Notified user's manager and HR per policy. Files were work-related but access was unauthorized."
            ),
        ],
        "Vulnerability": [
            (
                "Security scan found critical vulnerability",
                "Our weekly vulnerability scan flagged a critical CVE on the {product} server. CVE score is 9.8. Vendor has a patch available.",
                "Assessed vulnerability impact — server is internet-facing, so risk is high. Applied vendor patch during emergency maintenance window. Rescanned to confirm vulnerability remediated. Updated vulnerability tracking database."
            ),
            (
                "SSL certificate expiring soon",
                "Our SSL certificate for the customer portal expires in 5 days. Need to renew it before it expires or customers will get security warnings.",
                "Renewed SSL certificate through our certificate provider. Installed new certificate on the web server and load balancer. Verified certificate chain is complete and no mixed content warnings. Set up automated renewal reminders for 30 days before expiration."
            ),
            (
                "Need security review for new application",
                "We're about to deploy a new customer-facing {product} application. Security team requires a review before go-live. Can you schedule the assessment?",
                "Completed security assessment: penetration test, code review, and configuration audit. Found 2 medium-severity issues (CORS misconfiguration and missing rate limiting). Development team fixed both. Re-tested and approved for production deployment."
            ),
            (
                "Outdated software with known vulnerabilities",
                "We're still running an old version of {product} that has multiple known CVEs. We've been delaying the upgrade but security is now mandating it.",
                "Planned and executed upgrade to latest version during weekend maintenance window. All known CVEs addressed. Verified application functionality with automated test suite post-upgrade. No issues found. Updated CMDB with new version information."
            ),
        ],
        "Policy": [
            (
                "Need to encrypt laptop for compliance",
                "My laptop needs to have full disk encryption enabled for the upcoming compliance audit. Currently it's not encrypted. How do I get this done?",
                "Enabled BitLocker full disk encryption on user's Windows laptop. Recovery key stored in Active Directory. Encryption completed in background — took about 3 hours. Verified encrypted status for compliance records."
            ),
            (
                "Data classification help needed",
                "We're working with a new dataset in {department} and need help classifying it per our data handling policy. It contains customer names and email addresses but no financial data.",
                "Reviewed dataset contents against data classification policy. Classified as 'Internal - PII' level. Applied appropriate handling requirements: encrypted storage, access logging, no external sharing without DPA. Updated data inventory register."
            ),
            (
                "Security training overdue for team",
                "My entire team of 12 in {department} is overdue on their annual security awareness training. Can you send them the enrollment links?",
                "Sent enrollment invitations for security awareness training to all 12 team members with a completion deadline of 2 weeks. Set up automated reminders at 7-day and 3-day marks. Will report completion status to manager."
            ),
        ],
    },
    "Cloud/Infrastructure": {
        "VM": [
            (
                "VM not responding — can't SSH/RDP",
                "Production VM prod-web-03 is not responding to SSH connections. Also can't ping it. The services it hosts are down. Need urgent help.",
                "VM had run out of disk space causing the OS to hang. Connected via hypervisor console, cleared log files consuming 95% of disk, and restarted services. Added disk space monitoring alert to prevent recurrence."
            ),
            (
                "Need new VM provisioned",
                "Need a new Linux VM provisioned for the {department} team's new {product} deployment. Specs: 8 vCPU, 32GB RAM, 500GB SSD. Production environment.",
                "Provisioned Ubuntu 22.04 VM with requested specs in production cluster. Configured networking, firewall rules, and monitoring. Joined to configuration management. Provided SSH access to requestor and added to inventory."
            ),
            (
                "VM performance degraded — high CPU",
                "Our application VM has been running at 95%+ CPU for the past 2 days. Response times are terrible. We haven't deployed anything new recently.",
                "Identified runaway cron job executing every minute instead of every hour due to a configuration typo. Fixed the cron schedule. CPU usage dropped to normal 30% range. Application response times back to normal."
            ),
            (
                "VM snapshot taking up too much storage",
                "We have old VM snapshots from 3 months ago that are consuming 800GB of storage. Can they be safely deleted? The VMs have been running fine.",
                "Reviewed snapshot chain — confirmed 3-month-old snapshots are safe to remove as no rollback is planned. Consolidated and deleted old snapshots, freeing 800GB. Set policy to auto-delete snapshots older than 14 days."
            ),
            (
                "Need to resize VM — running out of resources",
                "Our staging VM needs to be upgraded — currently 4 vCPU and 16GB RAM but we're constantly hitting resource limits during load testing.",
                "Scheduled VM resize during maintenance window. Shut down VM, increased to 8 vCPU and 32GB RAM, and restarted. Load tests now complete without resource exhaustion. Monitored for 24 hours — stable."
            ),
        ],
        "Storage": [
            (
                "S3 bucket permissions need updating",
                "The {product} S3 bucket needs its permissions updated — we need the ETL service role to have read/write access but currently it only has read.",
                "Updated IAM policy for ETL service role to include s3:PutObject and s3:DeleteObject permissions on the specified bucket. Applied principle of least privilege — access limited to the /data/ prefix only. Tested and confirmed write access works."
            ),
            (
                "Storage volume running out of space",
                "The /data volume on our production server is at 94% capacity. Growing about 5GB per day. Need to expand before it fills up.",
                "Expanded EBS volume from 500GB to 1TB online — no downtime required. Extended the filesystem to use new space. Current usage now at 47%. Also implemented log rotation for the application logs that were the primary growth driver."
            ),
            (
                "Need to set up cross-region backup",
                "Compliance requires us to have our production data backed up to a different geographic region. Currently everything is in us-east-1.",
                "Configured cross-region replication from us-east-1 to us-west-2 for production S3 buckets. Set up automated EBS snapshot copy to target region. Verified replication with test data. Updated disaster recovery documentation."
            ),
            (
                "Object storage access very slow",
                "Downloads from our cloud storage are extremely slow — files that should take seconds are taking minutes. Affecting our data pipeline processing times.",
                "Application was making individual API calls for thousands of small files instead of batch operations. Refactored to use multipart downloads and parallelized transfers. Also enabled transfer acceleration on the bucket. Download times improved by 10x."
            ),
        ],
        "Deployment": [
            (
                "Deployment failed — application not starting",
                "Latest deployment to production failed — the application isn't starting. Getting a 'port already in use' error in the logs. Need to roll back.",
                "Previous deployment's process wasn't fully terminated due to a graceful shutdown timeout. Killed orphaned process, freed the port, and redeployed. Application started successfully. Updated deployment script to include a process cleanup step."
            ),
            (
                "CI/CD pipeline broken",
                "Our Jenkins pipeline has been failing for the past 3 builds. Error is in the Docker build step — image won't build. Blocking all deployments to staging and production.",
                "Docker build was failing because base image tag 'latest' had been updated upstream with breaking changes. Pinned base image to specific version tag, fixed the dependency that broke, and rebuilt. Pipeline green again."
            ),
            (
                "Need to set up deployment pipeline for new service",
                "We have a new microservice that needs a CI/CD pipeline set up. It's a Python Flask app deployed to our Kubernetes cluster. Repository is already in GitHub.",
                "Created Jenkins pipeline with stages: lint, test, Docker build, push to registry, deploy to staging. Added Kubernetes deployment manifests. Configured webhook for automatic builds on push to main. Tested full pipeline end-to-end."
            ),
            (
                "Rolling update causing brief outage",
                "Every time we deploy, there's a 30-60 second window where the application returns 502 errors. We're doing rolling updates but something isn't right.",
                "Health check endpoint was returning 200 before the application was fully ready to serve traffic. Added readiness probe that checks database connectivity and cache warmup. Rolling updates now happen with zero downtime."
            ),
            (
                "Need to rollback last deployment",
                "The deployment we pushed an hour ago has a critical bug — users are seeing incorrect data. Need to roll back to the previous version immediately.",
                "Rolled back to previous container image tag via Kubernetes rollout undo. Service restored to last known good version within 2 minutes. Verified data integrity post-rollback. Development team investigating the bug before next deployment."
            ),
        ],
        "Service Outage": [
            (
                "Production service completely down",
                "Our customer-facing {product} application is completely down. Users getting 503 errors. Multiple teams reporting issues. Started about 15 minutes ago.",
                "Root cause: expired TLS certificate on the load balancer. Renewed certificate and installed on all load balancer nodes. Service restored within 20 minutes of escalation. Implemented certificate expiry monitoring to alert 30 days before expiration."
            ),
            (
                "Intermittent 500 errors in production",
                "Users are intermittently getting 500 errors on our {product} platform. About 10% of requests fail. No pattern to which requests fail — seems random.",
                "One of three application pods was unhealthy due to memory leak, causing failures when load balancer routed to it. Restarted the unhealthy pod and identified the memory leak in recent code change. Deployed fix. Error rate dropped to 0%."
            ),
            (
                "Database connection issues causing service degradation",
                "Our application is very slow and occasionally timing out. Database team says the DB is fine but our app logs show connection timeouts.",
                "Network security group rule change had inadvertently reduced the connection limit between the app subnet and database subnet. Reverted the rule change. Connection pool immediately recovered and application performance normalized."
            ),
            (
                "Third-party API dependency down",
                "Our payment processing is failing because the third-party payment API is returning errors. Customers can't complete purchases. Vendor hasn't posted any status updates.",
                "Enabled failover to backup payment provider while primary vendor resolves their outage. Queued failed transactions for retry once primary comes back. Notified customer support team to handle any customer inquiries. Primary vendor restored service after 3 hours — all queued transactions processed."
            ),
        ],
        "Scaling": [
            (
                "Application can't handle traffic spike",
                "We're running a promotion and traffic is 5x normal levels. The application is buckling — slow response times and intermittent errors. Need more capacity now.",
                "Immediately scaled application horizontally from 3 to 10 pods. Increased database connection pool. Enabled CDN caching for static assets. Response times normalized within 10 minutes. Set up auto-scaling rules to handle future traffic spikes automatically."
            ),
            (
                "Need auto-scaling configured for new service",
                "Our new {product} service needs auto-scaling configured. We expect variable traffic — low overnight but peaks during business hours. Currently running fixed at 2 instances.",
                "Configured horizontal pod autoscaler with CPU threshold at 70% and memory at 80%. Min replicas: 2, max: 8. Added custom metrics based on request queue length. Tested with load simulator — scaling responds within 60 seconds of threshold breach."
            ),
            (
                "Cost optimization — over-provisioned resources",
                "Our cloud bill jumped 40% last month. I suspect we have over-provisioned resources that we're paying for but not using.",
                "Audit found: 5 idle VMs (shut down, saving $1,200/mo), 3 oversized instances (rightsized, saving $800/mo), 2TB of unused EBS volumes (deleted, saving $200/mo). Total monthly savings: $2,200. Set up cost monitoring alerts."
            ),
            (
                "Load balancer not distributing traffic evenly",
                "Two of our four application servers are getting 80% of the traffic while the other two sit nearly idle. Load balancer should be distributing evenly.",
                "Load balancer was using sticky sessions with an overly long persistence timeout (24 hours). Reduced session persistence to 15 minutes and changed algorithm from source-IP to round-robin with least-connections. Traffic now distributed evenly across all four servers."
            ),
        ],
    },
}


# ── Helper functions ───────────────────────────────────────────────────────────

def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_date(start_year=2023, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def resolution_hours(priority):
    """Generate realistic resolution time based on priority."""
    ranges = {
        "Critical": (0.5, 8),
        "High": (2, 24),
        "Medium": (4, 72),
        "Low": (8, 168),
    }
    low, high = ranges[priority]
    return round(random.uniform(low, high), 1)


def satisfaction_rating(hours, priority):
    """Generate satisfaction rating correlated with resolution speed."""
    # faster resolution → higher satisfaction
    ranges = {
        "Critical": (0.5, 8),
        "High": (2, 24),
        "Medium": (4, 72),
        "Low": (8, 168),
    }
    low, high = ranges[priority]
    midpoint = (low + high) / 2
    if hours < midpoint * 0.5:
        return random.choices([5, 4, 3], weights=[50, 35, 15])[0]
    elif hours < midpoint:
        return random.choices([5, 4, 3, 2], weights=[25, 40, 25, 10])[0]
    else:
        return random.choices([4, 3, 2, 1], weights=[15, 35, 35, 15])[0]


def weighted_choice(options_weights):
    """Choose from dict of {option: weight}."""
    options = list(options_weights.keys())
    weights = list(options_weights.values())
    return random.choices(options, weights=weights, k=1)[0]


def fill_placeholders(text, context):
    """Replace placeholders in text with context values."""
    for key, value in context.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def compose_description(base_desc, context, priority="Medium"):
    """Build a full description with optional opener, context, urgency, and closing."""
    parts = []

    opener = random.choice(OPENERS)
    if opener:
        parts.append(opener)

    parts.append(fill_placeholders(base_desc, context))

    extra = random.choice(EXTRA_CONTEXT)
    if extra:
        parts.append(fill_placeholders(extra, context))

    attempt = random.choice(ATTEMPTS)
    if attempt:
        parts.append(attempt)

    # Inject urgency phrase — this is what makes priority learnable from text.
    # 80% chance of including it so the signal is strong but not 100% mechanical.
    if random.random() < 0.80:
        parts.append(random.choice(URGENCY_PHRASES[priority]))

    closing = random.choice(CLOSINGS)
    if closing:
        parts.append(closing)

    return " ".join(parts)


# ── Main generation ────────────────────────────────────────────────────────────

def generate_tickets(num_tickets):
    tickets = []
    # Weight categories to get realistic distribution
    cat_weights = {
        "Network": 0.15,
        "Hardware": 0.18,
        "Software": 0.18,
        "Access/Permissions": 0.15,
        "Email/Communication": 0.12,
        "Database": 0.08,
        "Security": 0.07,
        "Cloud/Infrastructure": 0.07,
    }

    for i in range(num_tickets):
        ticket_id = f"TKT-{10001 + i}"

        # Pick category and subcategory
        category = weighted_choice(cat_weights)
        sub_categories = list(TEMPLATES[category].keys())
        sub_category = random.choice(sub_categories)

        # Pick a random template
        template_list = TEMPLATES[category][sub_category]
        subject_tmpl, desc_tmpl, resolution_tmpl = random.choice(template_list)

        # Build context for placeholder substitution
        cat_products = PRODUCTS_BY_CATEGORY[category]
        context = {
            "user": random_name(),
            "product": random.choice(cat_products),
            "error": random.choice(ERROR_CODES),
            "building": random.choice(BUILDINGS),
            "department": random.choice(DEPARTMENTS),
            "timeframe": random.choice(["yesterday", "this morning", "last week",
                                         "two days ago", "earlier today", "since Monday"]),
            "system": random.choice(cat_products),
        }

        subject = fill_placeholders(subject_tmpl, context)

        # Derive priority from subcategory so it correlates with ticket content.
        # Fall back to the global weights for subcategories not in the lookup.
        weights = PRIORITY_WEIGHTS_BY_SUBCATEGORY.get(sub_category, PRIORITY_WEIGHTS)
        priority = weighted_choice(weights)

        # Compose description AFTER priority is known so urgency phrases can be injected.
        description = compose_description(desc_tmpl, context, priority=priority)
        status = weighted_choice(STATUS_WEIGHTS)

        created = random_date()
        hours = resolution_hours(priority)

        if status in ("Closed", "Resolved"):
            resolved = created + timedelta(hours=hours)
            resolution = fill_placeholders(resolution_tmpl, context)
            rating = satisfaction_rating(hours, priority)
        else:
            resolved = None
            resolution = None
            rating = None
            hours = None

        department = random.choice(DEPARTMENTS)
        product_service = context["product"]

        tickets.append({
            "ticket_id": ticket_id,
            "created_date": created.strftime("%Y-%m-%d %H:%M"),
            "resolved_date": resolved.strftime("%Y-%m-%d %H:%M") if resolved else "",
            "category": category,
            "sub_category": sub_category,
            "priority": priority,
            "status": status,
            "subject": subject,
            "description": description,
            "resolution": resolution or "",
            "product_service": product_service,
            "department": department,
            "satisfaction_rating": rating if rating else "",
            "time_to_resolution_hours": hours if hours else "",
        })

    return tickets


def main():
    print(f"Generating {NUM_TICKETS} ITSM tickets...")
    tickets = generate_tickets(NUM_TICKETS)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fieldnames = [
        "ticket_id", "created_date", "resolved_date", "category", "sub_category",
        "priority", "status", "subject", "description", "resolution",
        "product_service", "department", "satisfaction_rating", "time_to_resolution_hours",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tickets)

    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Total tickets: {len(tickets)}")

    # Print distribution summary
    from collections import Counter
    cat_counts = Counter(t["category"] for t in tickets)
    pri_counts = Counter(t["priority"] for t in tickets)
    status_counts = Counter(t["status"] for t in tickets)

    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(tickets)*100:.1f}%)")

    print("\nPriority distribution:")
    for pri, count in sorted(pri_counts.items(), key=lambda x: -x[1]):
        print(f"  {pri}: {count} ({count/len(tickets)*100:.1f}%)")

    print("\nStatus distribution:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count} ({count/len(tickets)*100:.1f}%)")


if __name__ == "__main__":
    main()
