// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Switch
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.R
import helium314.keyboard.latin.media.provider.MediaPluginApprovalStore
import helium314.keyboard.latin.media.provider.MediaProviderClient
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.preferences.Preference
import helium314.keyboard.settings.preferences.PreferenceCategory

@Composable
fun MediaPluginsScreen(
    onClickBack: () -> Unit,
) {
    val context = LocalContext.current
    var refresh by remember { mutableIntStateOf(0) }
    val providers = remember(refresh) {
        val client = MediaProviderClient(context)
        try {
            client.getDiscoveredProviders()
        } finally {
            client.close()
        }
    }

    SearchSettingsScreen(
        onClickBack = onClickBack,
        title = stringResource(R.string.manage_media_plugins),
        settings = emptyList(),
        content = {
            Column {
                PreferenceCategory(stringResource(R.string.manage_media_plugins))
                if (providers.isEmpty()) {
                    Preference(
                        name = stringResource(R.string.media_plugins_no_plugins),
                        description = stringResource(R.string.media_plugins_no_plugins_summary),
                        onClick = {}
                    )
                } else {
                    providers.forEach { provider ->
                        val enabled = MediaPluginApprovalStore.isEnabled(context, provider)
                        fun setEnabled(value: Boolean) {
                            MediaPluginApprovalStore.setEnabled(context, provider, value)
                            KeyboardSwitcher.getInstance().setThemeNeedsReload()
                            refresh++
                        }
                        Preference(
                            name = provider.label,
                            description = provider.packageName + "/" + provider.serviceName,
                            onClick = { setEnabled(!enabled) }
                        ) {
                            Switch(
                                checked = enabled,
                                onCheckedChange = { setEnabled(it) },
                            )
                        }
                    }
                }
            }
        }
    )
}
