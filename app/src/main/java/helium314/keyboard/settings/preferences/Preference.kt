// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import androidx.annotation.DrawableRes
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.Hyphens
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import helium314.keyboard.latin.R
import helium314.keyboard.settings.IconOrImage
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.previewDark

// partially taken from StreetComplete / SCEE

@Composable
fun PreferenceCategory(
    title: String,
    modifier: Modifier = Modifier,
) {
    Column {
        HorizontalDivider()
        Text(
            text = title,
            modifier = modifier.padding(top = 12.dp, start = 16.dp, end = 8.dp, bottom = 8.dp),
            color = MaterialTheme.colorScheme.secondary,
            style = MaterialTheme.typography.titleSmall
        )
    }
}

@Composable
fun Preference(
    name: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    description: String? = null,
    @DrawableRes icon: Int? = null,
    value: @Composable (RowScope.() -> Unit)? = null,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clickable { onClick() }
            .heightIn(min = 44.dp)
            .padding(vertical = 10.dp, horizontal = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        if (icon != null)
            IconOrImage(icon, name, 32)
        Column(modifier = Modifier.weight(1f)) {
            Text(text = name, style = MaterialTheme.typography.bodyLarge)
            if (description != null) {
                CompositionLocalProvider(
                    LocalTextStyle provides MaterialTheme.typography.bodyMedium,
                    LocalContentColor provides MaterialTheme.colorScheme.onSurfaceVariant
                ) {
                    Text(
                        text = description,
                        modifier = Modifier.padding(top = 2.dp)
                    )
                }
            }
        }
        if (value != null) {
            CompositionLocalProvider(
                LocalTextStyle provides LocalTextStyle.current.copy(
                    textAlign = TextAlign.End,
                    hyphens = Hyphens.Auto
                ),
                LocalContentColor provides MaterialTheme.colorScheme.onSurfaceVariant
            ) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(
                        space = 8.dp,
                        alignment = Alignment.End
                    ),
                    verticalAlignment = Alignment.CenterVertically,
                ) { value() }
            }
        }
    }
}

@Preview
@Composable
private fun PreferencePreview() {
    Theme(previewDark) {
        Surface {
            Column {
                PreferenceCategory("Preference Category")
                Preference(
                    name = "Preference",
                    onClick = {},
                )
                Preference(
                    name = "Preference with icon",
                    onClick = {},
                    icon = R.drawable.ic_settings_about
                )
                SliderPreference(
                    name = "SliderPreference",
                    key = "",
                    default = 1,
                    description = { it.toString() },
                    range = -5f..5f
                )
                Preference(
                    name = "Preference with icon and description",
                    description = "some text",
                    onClick = {},
                    icon = R.drawable.ic_settings_about
                )
                Preference(
                    name = "Preference with switch",
                    onClick = {}
                ) {
                    Switch(checked = true, onCheckedChange = {})
                }
                SwitchPreference(
                    name = "SwitchPreference",
                    key = "none",
                    default = true
                )
                Preference(
                    name = "Preference",
                    onClick = {},
                    description = "A long description which may actually be several lines long, so it should wrap."
                ) {
                    Icon(painterResource(R.drawable.ic_arrow_left), null)
                }
                Preference(
                    name = "Long preference name that wraps",
                    onClick = {},
                ) {
                    Text("Long preference value")
                }
                Preference(
                    name = "Long preference name 2",
                    onClick = {},
                    description = "hello I am description"
                ) {
                    Text("Long preference value")
                }
            }
        }
    }
}
